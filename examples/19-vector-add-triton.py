"""GPU 向量加法教学实验。依赖 PyTorch + Triton + 受支持 GPU。
运行：python examples/19-vector-add-triton.py
本书制作环境未执行 GPU 测试；请在目标环境核验。
接口要求输出使用独立存储；保守拒绝与输入共享存储的所有非空切片。
"""
import statistics
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Z, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < n, other=0)
    y = tl.load(Y + i, mask=i < n, other=0)
    tl.store(Z + i, x + y, mask=i < n)


def launch(x, y, out):
    if x.device != y.device or x.device != out.device or x.device.type != 'cuda':
        raise ValueError('输入输出必须位于同一个 CUDA/ROCm 设备')
    if x.dtype != torch.float32 or y.dtype != x.dtype or out.dtype != x.dtype:
        raise ValueError('本实验仅支持 FP32')
    if x.ndim != 1 or x.shape != y.shape or x.shape != out.shape:
        raise ValueError('本实验仅支持相同长度的一维输入输出')
    if not (x.is_contiguous() and y.is_contiguous() and out.is_contiguous()):
        raise ValueError('本实验仅支持连续张量')
    if x.numel():
        # 比较整个底层存储的起始地址，而非切片 data_ptr：偏移切片也会被识别。
        # 教学接口保守拒绝同一存储，即便这些切片的实际元素范围没有重叠。
        out_storage = out.untyped_storage().data_ptr()
        if any(t.untyped_storage().data_ptr() == out_storage for t in (x, y)):
            raise ValueError('输出必须拥有独立存储，不允许原地或共享存储切片')
        add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('需要可用且受 Triton 支持的 CUDA/ROCm GPU；没有运行 CPU 冒充测试')
    torch.manual_seed(7)
    for n in (0, 1, 255, 257, 1000, 1_000_003):
        x = torch.randn(n, device='cuda', dtype=torch.float32)
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        launch(x, y, z)
        torch.testing.assert_close(z, x + y, rtol=1e-6, atol=1e-6)
    # 连续切片也可能部分重叠；必须在发射内核前拒绝。
    base = torch.arange(1026, device='cuda', dtype=torch.float32)
    x = base[:-1]
    y = torch.ones_like(x)
    for out in (x, base[1:]):
        try:
            launch(x, y, out)
        except ValueError:
            pass
        else:
            raise AssertionError('共享存储输出未被拒绝')
    print('向量加法边界、数值与共享存储拒绝检查通过')
    n = 1_000_003
    x, y = torch.randn(n, device='cuda'), torch.randn(n, device='cuda')
    z = torch.empty_like(x)
    for _ in range(20):
        launch(x, y, z)
    torch.cuda.synchronize()
    timings = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            launch(x, y, z)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / 100)
    ms = statistics.median(timings)
    print({'torch': torch.__version__, 'triton': triton.__version__,
           'gpu': torch.cuda.get_device_name(), 'shape': [n], 'dtype': 'float32',
           'BLOCK': 256, 'event_ms_median': ms,
           'logical_GB_s': 12 * n / (ms * 1e6),
           'mode': '同址重复普通启动，不含分配、复制和编译，可能受缓存与主机提交影响'})


if __name__ == '__main__':
    main()
