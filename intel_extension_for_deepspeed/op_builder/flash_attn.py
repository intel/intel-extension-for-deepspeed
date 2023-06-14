"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class FlashAttentionBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FlashAttention"
    NAME = "flash_atten"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/flash_attn/flash_attn.dp.cpp'),
            sycl_kernel_path('csrc/flash_attn/flash_attn_fwd.cpp'),
            sycl_kernel_path('csrc/flash_attn/flash_attn_bwd.cpp'),
        ]

    def include_paths(self):
        return [
            sycl_kernel_include('csrc/includes'),
            sycl_kernel_include('csrc/includes/flash_attn'),
            sycl_kernel_include('../../third_party/xetla/include'),
            'csrc/includes',
            '../../third_party/xetla/include',
            'csrc/includes/flash_attn',
        ]

    def extra_ldflags(self):
        args = super().extra_ldflags()
        args += ['-fsycl-targets=spir64_gen']
        args += ["-Xs \"-device pvc -options '-vc-disable-indvars-opt -vc-codegen -doubleGRF -Xfinalizer -printregusage -Xfinalizer -enableBCR -DPASTokenReduction '\" "]
        return args

    def cxx_args(self):
        args = super().cxx_args()
        args += ['-fsycl-targets=spir64_gen']
        return args
