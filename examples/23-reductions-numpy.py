"""NumPy FP64 参考：稳定 Softmax、RMSNorm 与输入梯度。
运行：python examples/23-reductions-numpy.py
"""
import numpy as np


def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_vjp(x, g):
    p = softmax(x)
    return p * (g - np.sum(g * p, axis=-1, keepdims=True))


def rmsnorm(x, w, eps=1e-6):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * r * w


def rmsnorm_dx(x, w, g, eps=1e-6):
    d = x.shape[-1]
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return r * g * w - x * r**3 / d * np.sum(g * w * x, axis=-1, keepdims=True)


def finite_difference(loss, x, h=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[idx] += h
        minus[idx] -= h
        grad[idx] = (loss(plus) - loss(minus)) / (2 * h)
    return grad


def main():
    rng = np.random.default_rng(7)
    max_error = 0.0
    for d in (1, 3, 7):
        x = rng.normal(size=(2, d)).astype(np.float64)
        w, g = rng.normal(size=d), rng.normal(size=x.shape)
        p = softmax(x)
        np.testing.assert_allclose(p.sum(-1), 1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p, softmax(x + 1000), rtol=1e-12, atol=1e-12)
        for func, grad in [(lambda z: softmax(z), softmax_vjp(x, g)),
                           (lambda z: rmsnorm(z, w), rmsnorm_dx(x, w, g))]:
            numeric = finite_difference(lambda z: float(np.sum(func(z) * g)), x)
            np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-7)
            max_error = max(max_error, float(np.max(np.abs(grad-numeric))))
    stable = softmax(np.array([[1000., 1001., 1002.]]))
    assert np.isfinite(stable).all()
    print({'gradient_checks': 6, 'max_abs_gradient_error': max_error,
           'scope': 'CPU FP64 数学参考验证，非 GPU 内核验证'})


if __name__ == '__main__':
    main()
