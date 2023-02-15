import torch
#import intel_extension_for_pytorch as ipex

data_type=torch.bfloat16

input = [ torch.tensor([[[i for i in range(1024)] for j in range(8)]], dtype=data_type),
        torch.tensor([[[i for i in range(1024)] for j in range(8)]], dtype=data_type),
        torch.tensor([[[i for i in range(1024)] for j in range(8)]], dtype=data_type)]
eps = 1e-5
input_layernorm = torch.nn.LayerNorm(1024, eps, dtype=data_type)
print (input_layernorm)
result = input_layernorm(input)
print (result)
