"""CPU 上验证 GEMM 的分块与尾部语义；不模拟 GPU 性能。
依赖 NumPy。运行：python examples/22-tiled-gemm-numpy.py
"""
import numpy as np


def tiled_gemm(a, b, bm=16, bn=16, bk=8):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('需要兼容的二维矩阵')
    if min(bm, bn, bk) <= 0:
        raise ValueError('块大小必须为正')
    # 教学参考使用 FP64 累加，避免把低精度误差与切块错误混淆。
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float64)
    for i in range(0, m, bm):
        for j in range(0, n, bn):
            for q in range(0, k, bk):
                c[i:i+bm, j:j+bn] += a[i:i+bm, q:q+bk] @ b[q:q+bk, j:j+bn]
    return c


def main():
    rng = np.random.default_rng(7)
    cases = [(37, 29, 41), (1, 7, 3), (17, 1, 18), (0, 3, 4), (3, 0, 4)]
    max_error = 0.0
    for m, k, n in cases:
        a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
        for blocks in [(16, 16, 8), (7, 5, 3)]:
            got = tiled_gemm(a, b, *blocks)
            ref = a @ b
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
            if got.size:
                max_error = max(max_error, float(np.max(np.abs(got-ref))))
    print({'cases': len(cases) * 2, 'max_abs_error': max_error,
           'scope': 'CPU FP64 数学分块验证，非 GPU 性能测试'})


if __name__ == '__main__':
    main()
