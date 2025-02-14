import torch
import torch.nn as nn

torch.manual_seed(420)

class MyModel(torch.nn.Module):

    def forward(self, input):
        permute = torch.tensor([0, 2, 1])
        x = input.permute(*permute)
        return x

input = torch.randn(2, 3, 4)

func = MyModel()
jit_func = torch.compile(func)

print(func(input))
# tensor([[[-0.0070,  0.0302,  0.5020],
#          [ 0.5044,  0.3826,  0.7538],
#          [ 0.6704, -0.5131,  0.6128],
#          [-0.3829,  0.7104, -0.9300]],
# 
#         [[-0.7392, -1.4816, -0.0021],
#          [ 0.4839,  0.3298, -1.1769],
#          [ 2.0201,  0.4856,  0.8036],
#          [-0.3333,  0.4131, -1.2524]]])

print(jit_func(input))
# torch._dynamo.exc.TorchRuntimeError