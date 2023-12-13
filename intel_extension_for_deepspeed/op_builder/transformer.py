"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class TransformerBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        return super().extra_ldflags()

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/onednn_wrappers.cpp'),
            sycl_kernel_path(
                'csrc/transformer/ds_transformer_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/onemkl_wrappers.cpp'),
            sycl_kernel_path('csrc/transformer/transform_kernels.cpp'),
            sycl_kernel_path('csrc/transformer/ds_gelu_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/gelu_kernels.cpp'),
            sycl_kernel_path('csrc/transformer/ds_dropout_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/dropout_kernels.cpp'),
            sycl_kernel_path(
                'csrc/transformer/ds_feedforward_sycl.cpp'),
            sycl_kernel_path(
                'csrc/transformer/ds_layer_reorder_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/ds_normalize_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/normalize_kernels.cpp'),
            sycl_kernel_path('csrc/transformer/ds_softmax_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/softmax_kernels.cpp'),
            sycl_kernel_path(
                'csrc/transformer/ds_stridedbatchgemm_sycl.cpp'),
            sycl_kernel_path('csrc/transformer/general_kernels.cpp')
        ]

    def include_paths(self):
        includes = [sycl_kernel_include('csrc/includes'), 'csrc/includes']
        return includes
