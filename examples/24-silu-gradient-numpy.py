"""SiLU 的 CPU FP64 有限差分验证。运行：python examples/24-silu-gradient-numpy.py"""
import numpy as np


def sigmoid(x):
    # 正负分支避免 exp 对极大正数/负数不必要地溢出。
    out = np.empty_like(x)
    mask = x >= 0
    out[mask] = 1 / (1 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    out[~mask] = exp_x / (1 + exp_x)
    return out


def main():
    x = np.array([-12., -3., -0.1, 0., 0.2, 3., 12.], dtype=np.float64)
    s = sigmoid(x)
    analytic = s + x * s * (1-s)
    h = 1e-6
    numerical = ((x+h) * sigmoid(x+h) - (x-h) * sigmoid(x-h)) / (2*h)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
    print({'max_abs_gradient_error': float(np.max(np.abs(analytic-numerical))),
           'scope': 'CPU FP64 公式验证'})


if __name__ == '__main__':
    main()
