#!/usr/bin/env python3
"""Logical uncompressed full-attention KV budget; not a device-memory measurement."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--element-bytes", type=float, default=2)
    args = parser.parse_args()
    if any(value <= 0 for value in vars(args).values()):
        parser.error("all parameters must be positive")
    per_token = 2 * args.layers * args.kv_heads * args.head_dim * args.element_bytes
    total = per_token * args.length * args.batch
    print(f"每token逻辑KV：{per_token / 1024:.3f} KiB")
    print(f"总逻辑KV：{total / 2**30:.3f} GiB")
    print("未计入块元数据、量化缩放/对齐、工作区、权重，也未计入分片或复制。")
    print("MLA、滑动窗口、状态空间或混合模型需要采用各自的缓存公式。")


if __name__ == "__main__":
    main()
