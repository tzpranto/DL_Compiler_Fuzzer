# https://github.com/pytorch/pytorch/issues/100988
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(420)

x = torch.randn(1, 3, 10, 10)

class Model(torch.nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=float('nan'))
        return x

func = Model().to('cpu')

jit_func = torch.compile(func)

res1 = func(x) # without jit
print(res1)
# succeed

res2 = jit_func(x)
print(res2)
# torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
# CppCompileError: C++ compile error