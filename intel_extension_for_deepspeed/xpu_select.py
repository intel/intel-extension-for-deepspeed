import os
import torch

# Choose XPU or CPU accelerator depending on XPU availability
def XPU_Accelerator():
    use_xpu = False
    use_cpu = True

    if use_xpu:
        import intel_extension_for_pytorch
        from .xpu_accelerator import _XPU_Accelerator
        xpu = _XPU_Accelerator()
        return xpu
    else:
        from .cpu_accelerator import _CPU_Accelerator
        cpu = _CPU_Accelerator()
        return cpu

