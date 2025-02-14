import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.functional.linear
        self.linear_weight = torch.randn(4, 4).cuda()
        self.bias = torch.randn(1, 4).cuda()

    def forward(self, x):
        x = self.linear(x, self.linear_weight, self.bias)
        return x

input_tensor = torch.randn(1, 3, 4).cuda()

func = Model().cuda()

res1 = func(input_tensor)
print(res1)
# tensor([[[-1.2507,  1.2743,  2.1668,  2.3092],
#          [ 0.2125,  0.0958, -2.3418,  3.3766],
#          [-0.3756,  0.8750, -0.5950,  4.4472]]], device='cuda:0')

jit_func = torch.compile(func)
res2 = jit_func(input_tensor)
# RuntimeError: The expanded size of the tensor (4) must match the existing size (2) at non-singleton dimension 0.  Target sizes: [4, 4].  Tensor sizes: [2, 4]
# While executing %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%l_x_, %l__self___linear_weight, %l__self___bias), kwargs = {})