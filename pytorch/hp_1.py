import torch

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.indices = [i for i in range(0, 10)]

    def forward(self, x):
        split_tensors = torch.split(x, 1, 0) # len(split_tensors) == 10
        chosen_tensors = [split_tensors[i] for i in self.indices if i in range(0, 10)]
        result = torch.cat(chosen_tensors, 0)
        return result
func = Model().to('cpu')

x = torch.randn(10, 3, 64, 64)

with torch.no_grad():
    func1 = torch.compile(func)
    print(func1(x.clone()).shape)
    # torch.Size([2, 3, 64, 64])
    print(torch._dynamo.utils.counters['inductor'])
    # Counter({'pattern_matcher_nodes': 11, 'pattern_matcher_count': 5, 'split_cat_norm': 2, 'cat_mutated': 1})

    print(func(x.clone()).shape)
    # torch.Size([10, 3, 64, 64])