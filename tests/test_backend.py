import deepspeed
from deepspeed.accelerator import get_accelerator

print (get_accelerator().device_name())
