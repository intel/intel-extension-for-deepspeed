import torch
import intel_extension_for_pytorch

a = torch.zeros(4)
print (a)
print (torch.xpu.is_available())
