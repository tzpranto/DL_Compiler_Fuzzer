import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        v1 = self.pool(self.conv2(x))
        v2 = self.pool(self.conv1(x))
        v3 = torch.mm(v1, v2)
        return v3


func = Model().to('cpu')
x = torch.randn(1, 3, 64, 64)
test_inputs = [x]

print(func(test_inputs))

jit_func = torch.compile(func)
print(jit_func(test_inputs))