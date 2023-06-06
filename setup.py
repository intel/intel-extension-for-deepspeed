from setuptools import setup
import subprocess
import os

version_str = "0.9.4"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
git_hash_cmd = "git rev-parse --short HEAD"


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


PACKAGE_NAME="intel_extension_for_deepspeed"

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")

if _check_env_flag("GIT_VERSIONED_BUILD", default="1"):
    version_str += f'+{git_hash}'

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
      license="https://opensource.org/license/mit")
