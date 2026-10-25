"""Triton 分块 FP16 GEMM 教学内核，FP32 累加。
依赖 PyTorch + Triton + 支持 FP16 dot 的 GPU；未在本书制作环境运行 GPU 验证。
运行：python examples/22-gemm-triton.py
目标是解释切块、步长与掩码，不承诺达到厂商库性能。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, M, N, K,
                SA0: tl.constexpr, SA1: tl.constexpr,
                SB0: tl.constexpr, SB1: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + rk
        a_ptrs = A + rm[:, None] * SA0 + kk[None, :] * SA1
        b_ptrs = B + kk[:, None] * SB0 + rn[None, :] * SB1
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm(a, b):
    if a.device != b.device or a.device.type != 'cuda':
        raise ValueError('需要同一个 CUDA/ROCm 设备')
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError('教学版本仅支持 FP16 输入')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('需要兼容的二维矩阵')
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if m and n:
        if k == 0:
            c.zero_()
        else:
            gemm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                BM=32, BN=32, BK=32, num_warps=4)
    return c


def main():
    if not torch.cuda.is_available():
        raise SystemExit('需要可用且受 Triton 支持的 CUDA/ROCm GPU')
    torch.manual_seed(7)
    for m, k, n in [(37, 29, 41), (1, 33, 7), (64, 64, 64), (0, 7, 3), (3, 0, 5)]:
        for transpose_b in (False, True):
            a = torch.randn((m, k), device='cuda', dtype=torch.float16)
            if transpose_b:
                b = torch.randn((n, k), device='cuda', dtype=torch.float16).T
            else:
                b = torch.randn((k, n), device='cuda', dtype=torch.float16)
            out = gemm(a, b)
            # FP64 参考避免默认 FP32 矩阵乘法精度选项影响参考值。
            ref = (a.double() @ b.double()).float()
            # 容差只服务本组小 K、标准正态样本，不能当作任意形状/尺度的精度保证。
            torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    print('GEMM 尾部、转置步长与数值检查通过；没有报告速度')


if __name__ == '__main__':
    main()
