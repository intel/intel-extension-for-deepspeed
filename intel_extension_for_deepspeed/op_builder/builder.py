"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import shutil
from pathlib import Path
from deepspeed.ops.op_builder.builder import OpBuilder, TORCH_MAJOR, TORCH_MINOR


class SYCLOpBuilder(OpBuilder):
    def builder(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DPCPPExtension

        print("dpcpp sources = {}".format(self.sources()))
        dpcpp_ext = DPCPPExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            extra_compile_args={
                'cxx': self.strip_empty_entries(self.cxx_args()),
            },
            extra_link_args=self.strip_empty_entries(self.extra_ldflags()))
        return dpcpp_ext

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        version_ge_1_3 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        version_ge_1_5 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def cxx_args(self):
        return ['-O3', '-g', '-std=c++20', '-w', '-fPIC', '-DMKL_ILP64']

    def extra_ldflags(self):
        return ['-fPIC', '-Wl,-export-dynamic']


def sycl_kernel_path(code_path):
    import intel_extension_for_pytorch
    abs_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    rel_path = os.path.join("third-party", code_path)
    print("Copying SYCL kernel file from {} to {}".format(abs_path, rel_path))
    os.makedirs(os.path.dirname(rel_path), exist_ok=True)
    shutil.copyfile(abs_path, rel_path)
    return rel_path


def sycl_kernel_include(code_path):
    import intel_extension_for_pytorch
    abs_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    return abs_path
