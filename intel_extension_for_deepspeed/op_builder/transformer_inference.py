# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class InferenceBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)

    def cxx_args(self):
        args = super().cxx_args()
        args.append('-DBF16_AVAILABLE')
        return args

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/inference/csrc/pt_binding.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/gelu.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/relu.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/layer_norm.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/rms_norm.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/softmax.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/dequantize.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/apply_rotary_pos_emb.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/transform.dp.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/pointwise_ops.dp.cpp'),
        ]

    def include_paths(self):
        includes = [
            sycl_kernel_include('csrc/transformer/inference/includes'),
            sycl_kernel_include('csrc/includes'),
        ]
        return includes

