"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import time
import importlib
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

    def load(self, verbose=True):
        from ...git_version_info import installed_ops, torch_info
        if installed_ops[self.name]:
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue. {self.error_log}"
            )
        try:
            import ninja  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        self.jit_mode = True
        from intel_extension_for_pytorch.xpu.cpp_extension import load

        start_build = time.time()
        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [
            self.deepspeed_src_path(path) for path in self.include_paths()
        ]

        # Torch will try and apply whatever CCs are in the arch list at compile time,
        # we have already set the intended targets ourselves we know that will be
        # needed at runtime. This prevents CC collisions such as multiple __half
        # implementations. Stash arch list to reset after build.
        '''
        torch_arch_list = None
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""
        '''

        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(sources),
            extra_include_paths=self.strip_empty_entries(extra_include_paths),
            extra_cflags=self.strip_empty_entries(self.cxx_args()),
            # extra_cuda_cflags=self.strip_empty_entries(self.nvcc_args()),
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
            verbose=verbose)
        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        '''
        # Reset arch list so we are not silently removing it for other possible use cases
        if torch_arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
        '''

        return op_module

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
