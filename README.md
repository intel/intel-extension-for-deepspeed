# Intel Extension for DeepSpeed
Intel Extension for DeepSpeed is an extension that brings Intel GPU (XPU) support to DeepSpeed(https://github.com/Microsoft/DeepSpeed).  It implements DeepSpeed Accelerator Interface as defined in https://github.com/microsoft/DeepSpeed/pull/2471.

Intel Extension for DeepSpeed comes with the following components:
1. DeepSpeed Accelerator Interface implementation
2. DeepSpeed op builders implmentation for XPU
3. DeepSpeed op builder kernel code

DeepSpeed would automatically use Intel Extension for DeepSpeed when it is installed as a python package.   After installation, models ported for DeepSpeed Accelerator Interface that run on DeepSpeed as in https://github.com/microsoft/DeepSpeed/pull/2471 could run on Intel GPU device.

## Usage:
### GPU:
1. Install Intel Extension for DeepSpeed

`python setup.py install`

2. Install DeepSpeed

`CC=dpcpp CFLAGS=-fPIC CXX=dpcpp CXXFLAGS=-fPIC DS_BUILD_DEVICE=dpcpp DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_QUANTIZER=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_UTILS=1 python setup.py install`

### CPU:
1.	Clone Intel Extension for DeepSpeed
```
git clone https://github.com/intel/intel-extension-for-deepspeed 
cd intel-extension-for-deepspeed
git checkout cpu-backend
python setup.py develop
```
2.  Clone DeepSpeed
```
git clone https://github.com/delock/DeepSpeedSYCLSupport
cd DeepSpeedSYCLSupport
git checkout gma/bf16_kernel
python -m pip install -r requirements/requirements.txt
python setup.py develop
```
3.  Install pytorch:
`python -m pip install pytorch`
4.	Install  ipex (CPU version):
`python -m pip install intel_extension_for_pytorch -f https://developer.intel.com/ipex-whl-stable-cpu`
5.	Install torch-ccl (CPU version)
`python -m pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu`
6.	Clone https://github.com/delock/Megatron-DeepSpeed branch: cpu-inference
```
git clone https://github.com/delock/Megatron-DeepSpeed
cd Megatron-DeepSpeed
git checkout cpu-inference
python -m pip install -r requirements.txt
```
7.	Under Megatron-DeepSpeed/dataset, download data with these commands:
```
bash download_vocab.sh
bash download_ckpt.sh     # remember install unzip before run this command
```
8.  copy gpt2-merges.txt, gpt2-vocab.json and checkpoints/ to Megatron-DeepSpeed directory (the parent directory of dataset/)
9.  Modify this line https://github.com/microsoft/Megatron-DeepSpeed/blob/main/tools/generate_samples_gpt.py#L160 to turn off kernel injection.
10.	Run this command from Megatron-DeepSpeed:
`echo Hiking in nature is | examples/generate_text.sh`

