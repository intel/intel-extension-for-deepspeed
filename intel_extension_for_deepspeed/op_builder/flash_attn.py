"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


flash_attn_module = None


class FlashAttnFunc(Function):

    @staticmethod
    def forward(ctx, query, key, value, dropout_p, softmax_scale, is_causal):
        """
        Shape of qkv and out: [Bs, Hn, Sl, Hs]
        Bs: batch size
        Hn: head number
        Sl: sequence length
        Hs: head size
        """
        bs, hn, sl, hs = query.shape
        if softmax_scale is None:
            softmax_scale = hs ** (-0.5)
        dropout_seed = torch.seed()
        is_training = True
        is_dropout = (dropout_p != 0)

        out, softmax_L = flash_attn_module.flash_attn_fwd(
            query, key, value, bs, hn, sl, hs, softmax_scale,
            dropout_p, dropout_seed,
            is_causal, is_training, is_dropout
        )

        ctx.save_for_backward(query, key, value, out, softmax_L)
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.is_dropout = is_dropout

        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_L = ctx.saved_tensors
        bs, hn, sl, hs = q.shape

        dq, dk, dv = flash_attn_module.flash_attn_bwd(
            dout, q, k, v, out, bs, hn, sl, hs, ctx.softmax_scale,
            ctx.dropout_p, ctx.dropout_seed,
            ctx.is_causal, ctx.is_dropout, softmax_L
        )
        return dq, dk, dv, None, None, None


class FlashAttentionBuilderObject():
    def __init__(self):
        pass
    
    # general functions
    def flash_attn_func(self, q, k, v,
            dropout_p, softmax_scale, is_causal):
        if q.dtype is torch.bfloat16:
            return FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, is_causal)
        else:
            return self.flash_attn_fwd_func(q, k, v, dropout_p)

    # forward functions
    def flash_attn_fwd_func(self, q, k, v, dropout_p):
        hs_rsqrt_scale = q.shape[-1] ** (-0.5)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores * hs_rsqrt_scale

        triu_mask = (torch.triu(torch.ones_like(attention_scores), diagonal=1) == 1)
        attention_scores.masked_fill_(triu_mask, -torch.inf)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = nn.Dropout(dropout_p)(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        return context_layer



class FlashAttentionBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FlashAttention"
    NAME = "flash_attn"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/flash_attn/flash_attn.dp.cpp'),
            sycl_kernel_path('csrc/flash_attn/fmha_fwd.cpp'),
            sycl_kernel_path('csrc/flash_attn/fmha_bwd.cpp'),
        ]

    def include_paths(self):
        return [
            sycl_kernel_include('csrc/includes'),
            sycl_kernel_include('csrc/flash_attn'),
            sycl_kernel_include('../../third_party/xetla/include'),
            'csrc/includes',
            'csrc/flash_attn',
            '../../third_party/xetla/include',
        ]

    def extra_ldflags(self):
        args = ['-fsycl', '-fPIC', '-Wl,-export-dynamic']
        args += ['-fsycl-targets=spir64_gen']
        args += ["-Xs \"-device pvc -options '-vc-disable-indvars-opt -vc-codegen -doubleGRF -Xfinalizer -printregusage -Xfinalizer -enableBCR -DPASTokenReduction '\" "]
        return args

    def cxx_args(self):
        args = ['-fsycl', '-O3', '-std=c++20', '-w', '-fPIC', '-DMKL_ILP64']
        args += ['-fsycl-targets=spir64_gen']
        args += ["-Xs \"-device pvc -options '-vc-disable-indvars-opt -vc-codegen -doubleGRF -Xfinalizer -printregusage -Xfinalizer -enableBCR -DPASTokenReduction '\" "]
        return args

    def load(self):
        global flash_attn_module
        flash_attn_module = super().load()
        return FlashAttentionBuilderObject()
