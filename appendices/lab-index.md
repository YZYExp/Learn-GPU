# 实验索引与复现方法

实验按设备条件分层。没有 GPU 仍可验证布局、梯度、矩阵分块、在线 Softmax、检查点和容量估算；设备内核则需要匹配的 GPU 与工具链。CPU 数学模型只用于解释正确性，不能拿它的耗时推断 GPU 性能。

## 标准 Python 与 NumPy

| 文件 | 观察内容 | 依赖 |
|---|---|---|
| `examples/01-training-cpu.py` | 手算训练、解析梯度与有限差分 | 标准库 |
| `examples/02-layout-cpu.py` | 形状、步长与共享存储 | 标准库 |
| `examples/10-environment.py` | 环境与 PyTorch 探测 | 标准库，可选 PyTorch |
| `examples/12-memory-budget.py` | 训练常驻状态的假设账本 | 标准库 |
| `examples/22-tiled-gemm-numpy.py` | 分块矩阵乘法与尾块 | NumPy |
| `examples/23-reductions-numpy.py` | 稳定 Softmax、RMSNorm 与梯度 | NumPy |
| `examples/24-silu-gradient-numpy.py` | SiLU 的有限差分验证 | NumPy |
| `examples/25-roofline_estimator.py` | 假设带宽/算力的屋顶线估算 | 标准库 |
| `examples/27-online_softmax.py` | 分块统计合并与直接结果对照 | 标准库 |
| `examples/33-resume-demo.py` | 保存完整状态与丢失动量的差别 | 标准库 |
| `examples/35-kv-budget.py` | 不同 KV 头数、批次与精度的容量 | 标准库 |

在工程根目录执行，例如：

```bash
python examples/01-training-cpu.py
python examples/22-tiled-gemm-numpy.py
python examples/27-online_softmax.py
python examples/33-resume-demo.py
```

这些实验已在制作环境实际运行；断言检查的是数学或状态恢复关系，而不是硬件成绩。更完整的命令和输出摘要记录于 `editorial/verification.md`。

## 框架与编译

| 文件 | 观察内容 | 依赖与边界 |
|---|---|---|
| `examples/14-accumulation.py` | 等大微批次的梯度累积 | PyTorch，CPU即可 |
| `examples/15-fx-graph.py` | FX 图与运行结果 | PyTorch，CPU即可 |
| `examples/16-compile-shapes.py` | 图捕获与不同输入形状 | PyTorch及可用编译后端 |
| `examples/18-jax-boundaries.py` | JIT 的追踪边界 | JAX，CPU即可 |
| `examples/24-custom-op-pytorch.py` | 自定义算子、Fake、Autograd 与编译 | PyTorch；例子限定连续FP32/FP64 |
| `examples/mini_transformer_train.py` | 小型完整训练、累积、精度与恢复 | PyTorch，CPU或CUDA；详见贯穿实验附录 |

在 PyTorch 2.14.1+cpu 中，第14章梯度累积、第15章FX、第16章CPU编译与形状变化、第24章CPU自定义算子注册/梯度/编译，以及完整小Transformer训练均已实际通过。完整训练还检查了累积、重计算、CPU编译与拆段恢复；条件见 `editorial/cpu-runtime-validation.md`。框架版本、设备和编译配置会影响行为，这些CPU结果不代表JAX、CUDA、BF16或Triton设备内核也已验证。

## GPU 内核

| 文件 | 教学目标 | 重点检查 |
|---|---|---|
| `examples/19-vector-add-triton.py` | 第一个向量加法内核与计时 | 尾块、空输入、别名与事件计时 |
| `examples/22-gemm-triton.py` | FP16输入、FP32累加的分块GEMM | 非整齐形状、步长、尾块与误差 |
| `examples/23-row-softmax-triton.py` | 行归约与融合Softmax | 行宽、填充、稳定性与容差 |

这些 GPU 路径未在制作环境的真实 GPU 上执行。代码提供教学起点，不能当作已经认证的生产内核。先记录环境，再验证输入覆盖和输出误差；确认正确之后才测稳态时间。随书示例有意保持约束明确，例如向量加法保守拒绝输出与输入共享存储。

## 一份可比较的实验记录

至少写下硬件型号、驱动、框架与工具版本，输入形状和布局，精度、前向/反向范围，编译与预热安排，计时方法，峰值内存口径，以及参考实现和误差条件。比较训练还应记录有效 token 数、有效全局批次、更新步数与损失质量。

如果把相同的微批次从单次执行改成梯度累积，要确认损失归一化方式；如果改变序列长度，要检查新增 token 是否真实参与损失；如果改变后端，要确认掩码、Dropout 和数值语义。让两种实现完成同一件事，是实验比速度更先需要满足的条件。
