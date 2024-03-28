from setuptools import setup
import subprocess
import os
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

PACKAGE_NAME="intel_extension_for_deepspeed"

version_str = "0.9.4"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
git_hash_cmd = "git rev-parse --short HEAD"

def get_project_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.abspath(project_root_dir)

def get_csrc_dir(op_name=""):
    project_root_dir = os.path.join(get_project_dir(), PACKAGE_NAME + "/op_builder/csrc/" + str(op_name))
    return os.path.abspath(project_root_dir)

def get_xetla_dir():
    project_root_dir = os.path.join(get_project_dir(), "third_party/xetla/include")
    return os.path.abspath(project_root_dir)

def create_ext_modules(op_name=""):
    cpp_files = []
    include_dirs = []

    for path, dir_list, file_list in os.walk(get_csrc_dir(op_name)):
        for file_name in file_list:
            if file_name.endswith('.cpp'):
                cpp_files += [os.path.join(path, file_name)]
    for path, dir_list, file_list in os.walk(get_csrc_dir()):
        for file_name in file_list:
            if file_name.endswith('.hpp') or file_name.endswith('.h'):
                include_dirs += [path]
                break
    include_dirs += [get_xetla_dir()]
    cxx_flags = [
        '-fsycl', '-O3', '-std=c++20', '-w', '-fPIC', '-DMKL_ILP64',
        '-fsycl-targets=spir64_gen',
        "-Xs \"-device pvc -options '-vc-disable-indvars-opt -vc-codegen -doubleGRF -Xfinalizer -printregusage -Xfinalizer -enableBCR -DPASTokenReduction '\" "
    ]
    extra_ldflags = [
        '-fsycl', '-fPIC', '-Wl,-export-dynamic', '-fsycl-targets=spir64_gen',
        "-Xs \"-device pvc -options '-vc-disable-indvars-opt -vc-codegen -doubleGRF -Xfinalizer -printregusage -Xfinalizer -enableBCR -DPASTokenReduction '\" "
    ]
    print("cpp_files: " + str(cpp_files))
    print("include_dirs: " + str(include_dirs))
    ext_modules = [
        DPCPPExtension(name="fmha_module",
                       sources=cpp_files,
                       include_dirs=include_dirs,
                       extra_compile_args={'cxx': cxx_flags},
                       extra_link_args=extra_ldflags)
    ]
    return ext_modules

def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}',
                              stdout=subprocess.PIPE,
                              shell=True)
    return result.wait() == 0


if command_exists('git'):
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"


def _build_installation_dependency():
    install_requires = []
    install_requires.append("setuptools")
    return install_requires

def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["Y", "1"];

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")

if _check_env_flag("GIT_VERSIONED_BUILD", default="1"):
    version_str += f'+{git_hash}'

ext_modules = create_ext_modules("flash_attn")
cmdclass = {'build_ext': DpcppBuildExtension}

long_description = ""
currentdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(currentdir, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

setup(name=PACKAGE_NAME,
      version=version_str,
      description="Intel® Extension for DeepSpeed*",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/intel/intel-extension-for-deepspeed",
      author="Intel Corporation",
      install_requires=_build_installation_dependency(),
      include_package_data=True,
      packages=[PACKAGE_NAME],
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      license="https://opensource.org/license/mit")
