#!/usr/bin/env python3
"""第 1、3 章：标准库即可运行，观察前向、解析梯度、差分与更新。

运行：python3 examples/01-training-cpu.py
这是单个标量样本的教学实验，不是 Transformer，也不是 GPU 性能测试。
"""
import math


def loss(w: float, b: float, x: float = 2.0, y: float = 5.0) -> float:
    return (x * w + b - y) ** 2


def main() -> None:
    w, b, x, y, lr = 1.0, 0.0, 2.0, 5.0, 0.1
    prediction = x * w + b
    residual = prediction - y
    dw, db = 2 * residual * x, 2 * residual
    h = 1e-5
    fdw = (loss(w + h, b) - loss(w - h, b)) / (2 * h)
    fdb = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
    assert math.isclose(dw, fdw, rel_tol=1e-8)
    assert math.isclose(db, fdb, rel_tol=1e-8)
    print(f"更新前: w={w}, b={b}, 预测={prediction}, 损失={loss(w, b)}")
    print(f"解析梯度: dw={dw}, db={db}; 中心差分: dw={fdw:.6f}, db={fdb:.6f}")
    w, b = w - lr * dw, b - lr * db
    assert math.isclose(loss(w, b), 0.0, abs_tol=1e-20)
    print(f"更新后: w={w}, b={b}, 预测={x * w + b}, 损失={loss(w, b)}")


if __name__ == "__main__":
    main()
