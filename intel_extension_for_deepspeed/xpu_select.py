import os
import torch

# Choose XPU or CPU accelerator depending on XPU availability
def XPU_Accelerator():
    try:
        from .xpu_accelerator import _XPU_Accelerator
    except ImportError:
        pass

    try:
        from .cpu_accelerator import _CPU_Accelerator
    except ImportError:
        pass

    use_xpu = False
    use_cpu = False

    if hasattr(torch, 'xpu'):
        use_xpu = True
    elif hasattr(torch, 'cpu'):
        use_cpu = True

    if use_xpu:
        xpu = _XPU_Accelerator()
        return xpu
    else:
        cpu = _CPU_Accelerator()
        return cpu

