# Intel® Extension for DeepSpeed*
Intel® Extension for DeepSpeed* is an extension that brings Intel GPU (XPU) support to DeepSpeed(https://github.com/Microsoft/DeepSpeed). It comes with the following components:
1. DeepSpeed Accelerator Interface implementation
2. DeepSpeed op builders implmentation for XPU
3. DeepSpeed op builder kernel code

DeepSpeed would automatically use Intel® Extension for DeepSpeed* when it is installed as a python package.   After installation, models ported for DeepSpeed Accelerator Interface that run on DeepSpeed could run on Intel GPU device.

Usage:
1. Install Intel® Extension for DeepSpeed*

`python setup.py install`

2. Install DeepSpeed

`CC=dpcpp CFLAGS=-fPIC CXX=dpcpp CXXFLAGS=-fPIC DS_BUILD_DEVICE=dpcpp DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_QUANTIZER=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_UTILS=1 python setup.py install`
