"""Compare compiled output across two CPU shapes.
Requires PyTorch with a working torch.compile backend and host toolchain.
Run: TORCH_LOGS=graph_breaks,recompiles python examples/16-compile-shapes.py
Compilation is intentionally left to the installed default backend.
"""
import torch

def block(x, scale, bias):
    return torch.relu(x * scale + bias)

compiled = torch.compile(block)
torch.manual_seed(7)
for batch in (32, 48, 32):
    x, scale, bias = torch.randn(batch, 128), torch.randn(128), torch.randn(128)
    expected, actual = block(x, scale, bias), compiled(x, scale, bias)
    torch.testing.assert_close(expected, actual)
    print("correct shape:", tuple(actual.shape))
