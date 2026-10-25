#!/usr/bin/env python3
"""第 2 章：以同一 Python 列表模拟张量的形状、步幅与转置视图。

运行：python3 examples/02-layout-cpu.py
只解释寻址，不模拟 GPU 缓存、内存事务或真实性能。
"""


def materialize(storage, shape, stride, offset=0):
    rows, columns = shape
    return [
        [storage[offset + i * stride[0] + j * stride[1]] for j in range(columns)]
        for i in range(rows)
    ]


def main():
    storage = list(range(6))
    original = materialize(storage, (2, 3), (3, 1))
    transposed = materialize(storage, (3, 2), (1, 3))
    reshaped = materialize(storage, (3, 2), (2, 1))
    assert original == [[0, 1, 2], [3, 4, 5]]
    assert transposed == [[0, 3], [1, 4], [2, 5]]
    assert reshaped == [[0, 1], [2, 3], [4, 5]]
    print("原始 [2,3] 步幅 [3,1]:", original)
    print("转置 [3,2] 步幅 [1,3]:", transposed)
    print("重塑 [3,2] 步幅 [2,1]:", reshaped)
    print("同形状的转置与重塑，数值排列不同；底层列表没有移动。")


if __name__ == "__main__":
    main()
