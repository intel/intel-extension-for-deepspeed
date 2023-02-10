import os
import intel_extension_for_pytorch as ipex  # noqa: F401
import oneccl_bindings_for_pytorch  #noqa: F401

from xpu_accelerator import _XPU_Accelertor
from cpu_accelerator import _CPU_Accelertor

# Choose XPU or CPU accelerator depending on XPU availability
def XPU_Accelerator():
    xpu = _XPU_Accelerator()

    # Environment variable has higher priority than device installed on the system
    # This allows prebuild on a machine that is different from target machine
    force_xpu_backend = os.environ['IDEX_FROCE_XPU_BACKEND']
    force_cpu_backend = os.environ['IDEX_FROCE_CPU_BACKEND']

    if force_xpu_backend == '1' or (not force_cpu_backend == '1' and xpu.is_available()):
        print ("XPU backend selected")
        return xpu
    else:
        cpu = _CPU_Accelerator()
        print ("CPU backend selected")
        return cpu

