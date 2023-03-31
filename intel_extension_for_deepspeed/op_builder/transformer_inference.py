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
        # TODO: check SYCL environment
        return super().is_compatible(verbose)

    # def load(self, verbose=True):
    #     # TODO: remove temporary bypass
    #     return None

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/inference/csrc/softmax.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/gelu.cpp'),
        ]

    def extra_ldflags(self):
        return []

    def include_paths(self):
        includes = [sycl_kernel_include('csrc/transformer/inference/includes'), 'csrc/transformer/inference/includes']
        return includes

