# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class QuantizerBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/quantization/pt_binding.cpp'),
            sycl_kernel_path('csrc/quantization/fake_quantizer.dp.cpp'),
            sycl_kernel_path('csrc/quantization/quantize.dp.cpp'),
            sycl_kernel_path('csrc/quantization/quantize_intX.dp.cpp'),
            sycl_kernel_path('csrc/quantization/dequantize.dp.cpp'),
            sycl_kernel_path('csrc/quantization/swizzled_quantize.dp.cpp'),
            sycl_kernel_path('csrc/quantization/quant_reduce.dp.cpp'),
        ]

    def include_paths(self):
        return [sycl_kernel_include('csrc/includes')]
