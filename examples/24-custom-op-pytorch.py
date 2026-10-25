"""PyTorch 自定义算子注册教学实验，不是高性能实现。
依赖支持 torch.library.custom_op/register_autograd 的现代 PyTorch。
运行：python examples/24-custom-op-pytorch.py
已在 PyTorch 2.14.1+cpu 实际通过CPU注册、契约、梯度与编译比较；CUDA未执行。
接口仅接受 CPU/CUDA 上连续、strided 布局的 FP32/FP64 张量。
"""
import torch


def validate_input(x):
    # 真正计算与 Fake 路径共享同一份纯元信息契约，不读取张量数值。
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError('教学算子仅支持 FP32/FP64 实数输入')
    if x.layout != torch.strided or not x.is_contiguous():
        raise ValueError('教学算子仅支持连续 strided 张量')
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('教学算子仅声明 CPU/CUDA 设备支持')


@torch.library.custom_op('gpu_atlas::silu_demo', mutates_args=())
def silu_demo(x: torch.Tensor) -> torch.Tensor:
    validate_input(x)
    return x * torch.sigmoid(x)


@silu_demo.register_fake
def silu_demo_fake(x):
    validate_input(x)
    # 与真实实现保持同形、同 dtype、同 device 的独立连续输出。
    return torch.empty_like(x)


def setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    s = torch.sigmoid(x)
    return grad_output * (s + x * s * (1 - s))


torch.library.register_autograd(silu_demo, backward, setup_context=setup_context)


def main():
    torch.manual_seed(7)
    # 整数 Sigmoid 的类型提升不能用 empty_like 冒充：两条路径均明确拒绝。
    for dtype in (torch.int64, torch.bool, torch.complex64, torch.float16):
        invalid = torch.ones((3, 7), dtype=dtype)
        for call in (silu_demo, silu_demo_fake):
            try:
                call(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('未拒绝未支持的数据类型')
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        # FP32 另做契约与输出检查；数值 gradcheck 使用 FP64。
        x32 = torch.randn((3, 7), device=device, dtype=torch.float32, requires_grad=True)
        torch.library.opcheck(silu_demo, (x32,))
        torch.testing.assert_close(silu_demo(x32), torch.nn.functional.silu(x32))
        x = torch.randn((3, 7), device=device, dtype=torch.float64, requires_grad=True)
        torch.library.opcheck(silu_demo, (x,))
        assert torch.autograd.gradcheck(silu_demo, (x,))
        torch.testing.assert_close(silu_demo(x), torch.nn.functional.silu(x))
        # 独立检查编译集成；需要适合该环境的编译后端与工具链。
        compiled = torch.compile(lambda z: silu_demo(z).sum(), fullgraph=True)
        got = compiled(x)
        gx, = torch.autograd.grad(got, x)
        ref = torch.nn.functional.silu(x).sum()
        rx, = torch.autograd.grad(ref, x)
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(gx, rx)
        print(device, '注册、梯度与编译比较通过；未作加速承诺')


if __name__ == '__main__':
    main()
