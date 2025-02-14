# https://github.com/pytorch/pytorch/issues/98979
import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2).to(torch.float64)

    def forward(self, x):
        x = x.permute(1, 0)
        x = self.linear(x)
        x = x.permute(1, 0)
        return x
input_tensor = torch.rand(2, 2).to(torch.float64)

func = Model().to('cpu')

print(func(input_tensor))
# tensor([[-1.0019, -0.4457],
#         [ 0.1512, -0.4111]], dtype=torch.float64, grad_fn=<PermuteBackward0>)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
    # torch._dynamo.exc.BackendCompilerFailed: backend='debug_wrapper' raised:
    # RuntimeError: dense_to_mkldnn expects float or bfloat16 tensor input