"""Inspect an FX symbolic graph; this is not the full torch.compile pipeline.
Requires PyTorch; CPU sufficient.
"""
import torch
from torch.fx import symbolic_trace

class AffineRelu(torch.nn.Module):
    def forward(self, x, scale, bias):
        return torch.relu(x * scale + bias)

module = AffineRelu()
graph_module = symbolic_trace(module)
x, scale, bias = torch.randn(2, 4), torch.randn(4), torch.randn(4)
torch.testing.assert_close(module(x, scale, bias), graph_module(x, scale, bias))
print(graph_module.graph)
