"""单程序处理一行的 Softmax 教学内核。
依赖 PyTorch + Triton + 受支持 GPU。仅连续二维 FP32、有限输入、1<=列数<=4096。
运行：python examples/23-row-softmax-triton.py
本书制作环境未执行 GPU 测试。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, mask=col < N, other=-float('inf'))
    z = x - tl.max(x, axis=0)
    numerator = tl.exp(z)
    out = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * N + col, out, mask=col < N)


def row_softmax(x):
    if x.device.type != 'cuda' or x.dtype != torch.float32:
        raise ValueError('需要 CUDA/ROCm 上的 FP32 输入')
    if x.ndim != 2 or not x.is_contiguous() or not 1 <= x.shape[1] <= 4096:
        raise ValueError('需要连续二维输入，列数在 1..4096')
    # 有限性检查属于教学输入校验，会同步设备，不能纳入内核性能比较。
    if not torch.isfinite(x).all().item():
        raise ValueError('本实验未定义无穷/NaN或全屏蔽行语义')
    y = torch.empty_like(x)
    if x.shape[0]:
        softmax_kernel[(x.shape[0],)](x, y, N=x.shape[1],
                                    BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=4)
    return y


def main():
    if not torch.cuda.is_available():
        raise SystemExit('需要可用且受 Triton 支持的 CUDA/ROCm GPU')
    torch.manual_seed(7)
    for rows, cols in [(0, 7), (1, 1), (5, 33), (17, 1000), (3, 4096)]:
        x = torch.randn((rows, cols), device='cuda', dtype=torch.float32)
        got = row_softmax(x)
        ref = torch.softmax(x, dim=-1)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(got.sum(-1), torch.ones(rows, device='cuda'),
                                   rtol=1e-4, atol=1e-6)
    print('Softmax 边界与数值检查通过；没有报告未经测量的加速比')


if __name__ == '__main__':
    main()
