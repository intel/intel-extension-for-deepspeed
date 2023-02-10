import os
import torch
import intel_extension_for_pytorch as ipex

from .xpu_accelerator import _XPU_Accelerator
from .cpu_accelerator import _CPU_Accelerator

# Choose XPU or CPU accelerator depending on XPU availability
def XPU_Accelerator():
    ipex_for_xpu = False
    ipex_for_cpu = False

    if hasattr(ipex, 'xpu'):
        ipex_for_xpu = True
    elif hasattr(ipex, 'cpu'):
        ipex_for_cpu = True

    print (ipex_for_xpu)
    print (ipex_for_cpu)

    if ipex_for_xpu:
        xpu = _XPU_Accelerator()
        print ("XPU backend selected")
        return xpu
    else:
        cpu = _CPU_Accelerator()
        print ("CPU backend selected")
        return cpu

