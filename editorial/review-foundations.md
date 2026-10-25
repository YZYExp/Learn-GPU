# 交叉审稿：第 13—18、25—30 章

审稿日期：2026-10-25。审稿人：foundations。仅阅读，没有修改被审章节与示例。

## 结论

未发现需要阻止整合的严重公式错误、现代API过时误用或虚构性能实测。正文已较好地区分数学估算、实现约束和未实测部分。找到两处值得在发布前直接修正的小问题。

## 必要修订

### P2：第 13 章的“二位有效数字+共享缩放”练习，没有足够条件体现缩放与异常值的关系

位置：`chapters/13-low-precision.md:56`。

现文只规定“保留两位有效数字”，没有规定有限指数范围、下溢或饱和。在这种理想十进制浮点模型中，给所有值按十的幂共同缩放，再恢复，并不会自然展示异常大值挤压小值表示范围。`1.01、1.02、1.03` 都舍入到 `1.0`、`1000` 可照常表示，读者容易把“有效数字舍入”与“有限动态范围下共享缩放”误当同一机制。

最小修改：把练习拆成两问，先用两位有效数字观察舍入，再用一个**明确标为整数玩具量化、非FP8编码**的模型理解共享缩放，例如 `q=clip(round(x/s), -7, 7)`。对 `[0.11,0.21,0.51,7]` 比较全组 `s=1` 与小值单独成块、使用 `s=0.1` 的恢复误差。这样约束完整，也能看出缩放粒度的作用。若坚持浮点练习，则补上假想格式的最小/最大指数以及下溢处理规则。

### P2：第 14 章把仅比较梯度的示例称为“比较更新”

位置：`chapters/14-memory-stability.md:58`；相关代码 `examples/14-accumulation.py:17`。

示例对大批量和微批次分别 `backward()`，然后比较 `.grad`，没有创建优化器、调用 `step()` 或比较更新后的参数。正文“比较一次大批量更新与等大微批次累积”稍微扩大了实际验证范围。

最小修改：把这句话改为“比较一次大批量计算与等大微批次累积得到的参数梯度”。也可以保留当前文字，给两个模型用相同SGD执行一次更新并再比较参数；前一做法更轻量。

## 已检查的关键点

- 第 13 章 BF16/FP16、autocast/GradScaler分离、FP8缩放、FP4配方与状态账本均合理。NVFP4论文链接 `arXiv:2509.25149` 已实际打开，标题与内容相符。
- 第 14 章非重入检查点、随机状态、梯度累积归一化及先unscale后clip的顺序合理。
- 第 15—16 章求导图/编译图/CUDA Graph的边界清楚；`dynamic=None/True/False` 与 `fullgraph=True` 的描述对照当前PyTorch官方文档一致。
- 第 17 章 `24N→8N` 字节模型、4MiB转置最低8MiB有效载荷等计算正确，且明确没有承诺实际3倍加速。
- 第 18 章JAX追踪、StableHLO、MLIR与异步边界没有发现明显误导。
- 第 25 章Amdahl、Roofline、FP32向量加法83.3GFLOP/s教学计算正确。MFU与重计算区分合理。
- 第 26 章双缓冲生命周期、3+4×5=23时间模型和片上缓冲容量估算正确；明确硬件差异与未实测。
- 第 27 章在线Softmax统计量、50/3手算、N²中间张量大小正确。FA4官方原文已打开，描述与其硬件瓶颈/流水线/CuTe DSL内容相符。
- 第 28 章600秒/1微秒=6亿次调用的摊销计算正确；CuTe DSL与cuTile明确区分，后端成熟度未夸大。Helion调优文档与CUDA Tile文档均实际打开。
- 第 29 章集合语义、环形每Rank发送量、8卡1GB→1.75GB→35ms简化估算正确，限制条件已写明。
- 第 30 章ZeRO状态账本、FSDP2接口、DTensor与模块通信边界描述合理；官方 `fully_shard` 文档链接已查询并确认存在。

## 验证证据与边界

- 本审稿实际执行 `examples/25-roofline_estimator.py`，工作示例断言通过。
- 本审稿实际执行 `examples/27-online_softmax.py`，前向检查通过，最大绝对误差约1.11e-16。
- 所审章节的Python代码块通过AST语法解析。
- 没有在本审稿运行PyTorch/JAX或GPU代码，不能将这些检查解释为其运行验证。

## 本次额外核查的官方资料

- PyTorch Dealing with Recompilations：<https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html>
- PyTorch FSDP2 fully_shard：<https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>
- NVIDIA等，Pretraining Large Language Models with NVFP4：<https://arxiv.org/abs/2509.25149>
- Tri Dao，FlashAttention-4官方博客：<https://tridao.me/blog/2026/flash4/>
- Helion Deployment and Autotuning：<https://helionlang.com/deployment_autotuning.html>
- NVIDIA Writing Tile Kernels：<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html>
