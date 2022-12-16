# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.accelerator.real_accelerator import get_accelerator


FLOAT_TYPES = (torch.FloatTensor, get_accelerator().FloatTensor)
HALF_TYPES = (torch.HalfTensor, get_accelerator().HalfTensor)

def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn =  [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

def fp32_to_bf16(val):
    """Convert fp32 `val` to bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.bf16()
        return val
    return conversion_helper(val, half_conversion)

def bf16_to_fp32(val):
    """Convert bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)

class BF16_Module(nn.Module):
    def __init__(self, module):
        super(BF16_Module, self).__init__()
        self.add_module('module', module.bfloat16())

    def forward(self, *inputs, **kwargs):
        return bf16_to_fp32(self.module(*(fp32_to_bf16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

