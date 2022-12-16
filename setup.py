from setuptools import setup
import subprocess

version_str = "1.0"
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

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
version_str += f'+{git_hash}'

setup(name="intel_extension_for_deepspeed",
      version=version_str,
      description="Intel Extension for DeepSpeed",
      author="Intel Corporation",
      include_package_data=True,
      packages=["intel_extension_for_deepspeed"])
