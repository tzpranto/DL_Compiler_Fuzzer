# https://github.com/pytorch/pytorch/issues/98852

import torch
import torch.nn as nn

torch.manual_seed(420)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.f = nn.Sequential(nn.Linear(10, 20), self.relu, nn.Linear(20, 30), self.relu, nn.Linear(30, 40), self.relu)

    def forward(self, x):
        x = self.f(x)
        return x
input_tensor = torch.randn(2, 10)

func = Net()
jit_func = torch.compile(func)

print(func(input_tensor))
print(jit_func(input_tensor))