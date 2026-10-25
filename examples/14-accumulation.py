"""CPU example: equal microbatches reproduce a linear model's mean gradient.
Requires PyTorch. No GPU performance claim. Run from repository root.
"""
import copy
import torch

torch.manual_seed(7)
base = torch.nn.Linear(4, 1).double()
large = copy.deepcopy(base)
micro = copy.deepcopy(base)
x = torch.randn(16, 4, dtype=torch.float64)
y = torch.randn(16, 1, dtype=torch.float64)
loss_fn = torch.nn.MSELoss(reduction="mean")
loss_fn(large(x), y).backward()
for xs, ys in zip(x.chunk(4), y.chunk(4)):
    (loss_fn(micro(xs), ys) / 4).backward()
for p_large, p_micro in zip(large.parameters(), micro.parameters()):
    torch.testing.assert_close(p_large.grad, p_micro.grad, rtol=1e-10, atol=1e-12)
print("Equal-size microbatch gradients agree within float64 tolerance.")
