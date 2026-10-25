# CPU 运行验证记录

日期：2026-10-25。此次验证补充此前的源码审阅和数学参考检查；没有使用或验证真实 GPU。

## 环境与隔离

- Linux x86_64，Python 3.12.14，PyTorch **2.14.1+cpu**。
- 通过官方 CPU 索引安装：`python -m pip install --index-url https://download.pytorch.org/whl/cpu torch`。未安装 CUDA PyTorch 发行包或 NVIDIA CUDA 依赖。
- 独立环境：`build/cpu-venv`，使用 `--system-site-packages` 创建。基础 Python 环境没有因此获得 torch。
- 子进程固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。CPU 编译使用当前默认 TorchInductor 后端，主机存在 `g++`；`TORCHINDUCTOR_COMPILE_THREADS=1`，缓存放在 `build/cpu-validation/inductor-cache`。
- `torch.cuda.is_available()` 为 False。CUDA、BF16、FP16、Triton、ROCm、JAX 与多卡路径均未因此获得运行验证。
- venv、编译缓存、检查点、原始运行日志均在忽略目录 `build/`，不应进入工程包或 Git。

## 完整 Transformer 实验

被测程序：`examples/mini_transformer_train.py`。公共极小配置：

```text
--device cpu --batch 2 --seq 8 --dim 16 --heads 2 --layers 1 --vocab 16
precision=fp32, lr=0.002, seed=20261025
```

该模型实际有3952个参数，使用因果SDPA、PreNorm、MLP和AdamW。任务是合成的循环整数下一词元预测，不是真实语言数据。

| 检查 | 在公共配置上增加的选项 | 实际结果 |
|---|---|---|
| 普通连续训练 | `--steps 4 --checkpoint build/cpu-validation/continuous.pt` | 完成4次更新、输出有限损失、保存成功，退出0 |
| 分段首段 | `--steps 2 --save-every 1 --checkpoint build/cpu-validation/split.pt` | 在完整更新后保存，退出0 |
| 新进程恢复 | `--resume build/cpu-validation/split.pt --steps 4`，不再重传公共配置 | 正确继承配置，从第3步继续至第4步，退出0 |
| 梯度累积 | `--steps 3 --accum 2` | 完成3次更新与保存，退出0 |
| 激活重计算 | `--steps 3 --activation-checkpoint` | 非重入检查点完整前后向通过，退出0 |
| 累积加重计算 | `--steps 3 --accum 2 --activation-checkpoint` | 两种机制组合通过，退出0 |
| CPU编译训练 | `--steps 2 --compile` | 默认Inductor的前后向、更新与保存通过，退出0 |

普通连续训练的逐步损失为：

```text
step 1: 2.8384149074554443
step 2: 2.9204752445220947
step 3: 2.5812015533447266
step 4: 2.4854209423065186
```

这只是功能烟雾实验，不据四步曲线声明长期收敛。编译训练前两步损失与此记录相同；这没有证明任意形状、精度或模型编译后逐位一致。

### 恢复验收：逐字段精确比较

用 `torch.load(..., map_location='cpu', weights_only=True)` 加载连续四步和拆段恢复后的最终文件，对字典、列表、元组递归比较，所有张量使用 `torch.equal`：

- 模型的全部参数与缓冲区：相等。
- AdamW的全部状态、参数组及计数：相等。
- 已完成步数：均为4；完成更新标识为True。
- 模型与训练配置：相等。
- 独立合成数据Generator状态：相等。
- PyTorch CPU RNG、Python RNG：相等；CUDA RNG在该CPU运行中均为None。
- 格式版本、设备和PyTorch版本等其他保存字段：相等。

因此，**在上述CPU配置与当前运行条件下，拆段恢复后的完整检查点与连续运行逐字段精确一致**。比较的是加载后的状态，不是两个ZIP文件的字节，因为序列化容器本身可能包含不同记录信息。这一结果不外推到CUDA、跨版本、跨设备或其他任务。

## 第14、15、16、24章框架实验

| 程序 | 实际执行内容 | 结果 |
|---|---|---|
| `examples/14-accumulation.py` | FP64线性模型，全批平均梯度与4个等大小微批次累积梯度，`rtol=1e-10, atol=1e-12` | 通过，退出0 |
| `examples/15-fx-graph.py` | FX符号图与原模块输出比较，打印乘法/加法/ReLU节点 | 通过，退出0 |
| `examples/16-compile-shapes.py` | 默认CPU Inductor，对32/48/32批次形状逐次比较编译与未编译输出 | 三个形状均通过，退出0 |
| `examples/24-custom-op-pytorch.py` | CPU FP32/FP64算子契约、Fake与真实路径拒绝不支持dtype、opcheck、FP64 gradcheck、fullgraph编译输出与输入梯度对照 | 全部通过，退出0；CUDA分支未进入 |

第1—3章全部内嵌Python代码也已实际执行：第1章小线性训练完成，第2章形状/步幅/连续性输出符合正文，第3章参数梯度输出24。

未将CPU自定义算子注册成功解释为高性能实现，也没有把CPU编译成功解释为GPU编译通过。

## 计时记录的解释

所有命令退出0。原始stdout/stderr存于 `build/cpu-validation/`，汇总为 `results.json`，恢复精确比较为 `exact-resume.json`。以下仅为本次验证子进程的主机墙钟时长，包含Python导入、编译（如有）、训练及保存；不是设备跑分：

| 验证命令 | 本次进程墙钟秒数 |
|---|---:|
| 连续四步 | 2.365 |
| 分段前两步 | 2.313 |
| 新进程恢复 | 2.182 |
| 累积 | 2.195 |
| 重计算 | 2.263 |
| 累积与重计算 | 2.319 |
| 第14章 | 1.726 |
| 第15章 | 1.684 |
| 第16章CPU编译 | 16.529 |
| 第24章CPU注册与编译 | 5.573 |
| 小Transformer CPU编译 | 13.659 |

CPU编译小模型的首次更新内部计时约10.075秒，第二次约2.161毫秒；只有两步，且共享了本次验证的编译缓存，不足以估计稳定吞吐、比较编译加速比或预测其他环境开销。记录这两个数只为确认首次准备与后续执行的时间边界确实被区分。

## 未覆盖项目

- 没有CUDA设备，没有执行BF16 autocast、GPU峰值显存统计、GPU保存恢复或CPU↔GPU迁移。
- 没有执行Triton内核、ROCm或多GPU通信。
- 没有运行JAX示例。
- 没有进行长时间训练、全形状回归、错误恢复压力测试或性能基准认证。
- 累积与重计算检查证明所测配置能完整执行；本次没有将它们各自的最终参数与其他batch组织逐位比较。

本次已据实更新贯穿实验附录、实验索引和相关来源记录中的验证状态；原有“未测GPU”的边界保留。
