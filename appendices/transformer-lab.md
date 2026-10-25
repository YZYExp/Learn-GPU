# 贯穿实验　从 CPU 到 GPU 的小型 Transformer 训练

本书各章反复出现“小型 Transformer”。这里把它落实为一份完整程序：`examples/mini_transformer_train.py`。它包含词元嵌入、位置嵌入、多头因果自注意力、残差、归一化、MLP、交叉熵、AdamW、梯度累积与恢复。输入到参数更新之间没有省略的训练步骤，也不需要下载数据集。

**验证状态：已在 Linux x86_64、Python 3.12.14、PyTorch 2.14.1+cpu 上实际运行极小模型，完成普通训练、梯度累积、激活重计算、二者组合及 CPU 编译训练检查。连续训练四步，与训练两步、退出后恢复到第四步的完整检查点逐字段一致，包括参数、优化器与全部保存的随机状态。CUDA、BF16 和 GPU 显存路径尚未执行。** 这验证了所测 CPU 配置的功能与恢复，不是模型质量或硬件性能认证；详细条件见工程中的 `editorial/cpu-runtime-validation.md`。

## 先知道模型在学什么

词表默认有 32 个整数。每条序列随机选择起点，之后按模 32 加一：例如输入 `7,8,9,10`，目标就是 `8,9,10,11`；走到 31 后回到 0。模型在每个位置预测下一个词元，因果注意力只允许读取当前位置及其之前的数据。[1]

这是合成规则学习。当前词元已足够决定答案，模型甚至可以主要依靠词嵌入与输出层完成任务；因此损失下降不能证明语言理解或长距离推理，也不能证明注意力结构不可替代。我们使用它，是为了获得固定形状、无下载、容易复现的训练计算骨架，观察显存、编译、重计算和计时。以后可以更换任务，但应重新说明验证目标。

模型采用 PreNorm：先归一化，再做注意力或 MLP，最后加回残差。Q、K、V 的形状为 `[batch, heads, sequence, head_dim]`；SDPA 调用使用 `is_causal=True` 和 `dropout_p=0.0`。位置嵌入最大长度就是配置的 `seq`，恢复时也不能随意改变。为减少随机因素，本例不包含 Dropout；它仍是完整可训练的小型 Decoder。

## 第一步：CPU 上检查完整循环

先安装适合自己环境的 PyTorch，再从工程根目录运行：

```bash
python3 examples/mini_transformer_train.py --device cpu --steps 20
```

默认 FP32、微批次 8、序列长度 32、隐藏宽度 64、4 个头、2 个块。更小的烟雾实验可以使用：

```bash
python3 examples/mini_transformer_train.py --device cpu --steps 3 --batch 2 --seq 8 --dim 32 --heads 4 --layers 1 --checkpoint build/smoke.pt
```

程序打印配置和参数量，再逐步输出损失、耗时、处理词元数与吞吐。每个 step 都表示一次完成的优化器更新。先检查输出是否有限、程序是否能完成、模型保存是否成功，再增加步数观察损失趋势。几步内没有明显下降，不能立即判断实现错误；反过来，也不能仅凭几步下降就认定训练完全正确。

## 第二步：保持输入不变，换到 CUDA

```bash
python3 examples/mini_transformer_train.py --device cuda --steps 20 --checkpoint build/cuda-fp32.pt
python3 examples/mini_transformer_train.py --device cuda --precision bf16 --steps 20 --checkpoint build/cuda-bf16.pt
```

每条命令都从相同种子重新开始，适合做配置对照。BF16 使用 autocast，参数与 AdamW 状态仍按该实现的默认 FP32 路径管理，不能把它叫作“所有张量都是 BF16”。程序只在支持 BF16 的 CUDA 设备上接受该选项；CPU 路径固定使用 FP32。[2]

本例不提供 FP16。添加 FP16 需要连同 GradScaler、取消缩放、裁剪和保存缩放器状态一起处理，不能只在参数中多列一个名称。FP32 CUDA 路径关闭可选 TF32 矩阵与 cuDNN 设置，避免让标签含义随环境默认值变化。不同设备上的结果仍不保证逐位一致。

## 第三步：编译与梯度累积分别做实验

```bash
python3 examples/mini_transformer_train.py --device cuda --precision bf16 --compile --steps 20 --checkpoint build/compiled.pt
python3 examples/mini_transformer_train.py --device cpu --batch 2 --accum 4 --steps 20 --checkpoint build/accumulated.pt
```

`--compile` 编译前向与损失函数，不把数据生成、日志和磁盘保存放入图中。autocast 上下文位于该函数内部，其外部调用与 `backward()` 都处于无 autocast 的环境，保持编译反向所假定的环境一致。当前编译器对反向 autocast 有专门约定；修改边界时应重新核查，不能把外部上下文随意挪动。[3]

首次更新可能包含追踪、编译和 AdamW 状态初始化，日志会标记 `first_update_in_process` 与 `may_include_compilation`。不要把这一条与稳态调用混成一个数字，也不要承诺所有后续步骤都没有延迟编译。编译依赖目标后端及工具链，CPU 同样可以尝试，但不保证小模型有收益。

`batch` 表示每个微批次的序列数，`accum` 表示一次更新累积几个微批次。示例中 `2×4=8`，与默认 `8×1` 的有效序列数相同；固定序列长度且没有填充，使每次损失除以 `accum` 后对应有效批次的平均梯度。先累积完，再统一裁剪与更新。比较不同组合时，还应保持总更新数、有效词元数和精度相同；浮点归约顺序变化仍可能产生差异。

若想体验激活重计算，可额外加入 `--activation-checkpoint`。它对每个 Decoder 块使用 `checkpoint(..., use_reentrant=False)`，与磁盘检查点 `--checkpoint` 是两回事。小模型可能看不出明显显存收益，新增重算反而容易表现为时间增长。[4]

## 第四步：真正退出，再恢复

```bash
python3 examples/mini_transformer_train.py --device cpu --steps 10 --save-every 5 --checkpoint build/resume.pt
python3 examples/mini_transformer_train.py --resume build/resume.pt --steps 20
```

第二条命令的 20 是**累计目标步数**，因此从 10 接着完成 11 到 20，而不是再跑 20 步。恢复时继承模型尺寸、批量、精度、累积次数、学习率、种子和编译等配置；如果显式传入冲突配置，会报错，避免把另一项实验误称为连续恢复。

检查点保存模型、AdamW 状态、已完成步数、配置、独立数据 Generator 状态，以及 Python、PyTorch CPU 和所用 CUDA 设备的随机状态。模型构造和加载结束后才恢复全局随机状态，防止初始化消耗恢复后的随机序列。数据起点由独立 CPU Generator 产生，因此编译或模型随机行为不会打乱后续数据顺序。

程序只在完整优化步骤结束并同步后保存，先写临时文件，再替换目标文件；不会把半组梯度累积保存为完成更新。若中途崩溃，只能从最后成功写入的检查点继续。`save-every=0` 表示正常结束时保存；它不是每一步都落盘。

验收时，可另外运行一次连续 20 步，将第 11—20 步损失和最终状态与分段恢复结果对照。相同设备、软件、配置和数据流是重要前提；CUDA 算法、编译与硬件差异仍可能引入非确定性。保存 RNG 保证我们具备恢复随机过程所需的信息，不等于宣称跨平台逐位相同。[5]

## 时间与显存该怎样读

每步计时在起点先等待设备，终点也同步；范围包含合成数据生成、搬运、全部微批次的前后向、裁剪和参数更新，不包含日志格式化与检查点 I/O。吞吐分子为 `batch×seq×accum` 个目标词元。因此它是本实验的同步更新吞吐，不是纯 GPU 内核吞吐，更不是自然语言训练跑分。

逐步同步便于理解计时边界，也会削弱真实流水线可能获得的重叠。CUDA 日志分别给出当步最大 allocated 和 reserved 显存；前者关注张量分配，后者还反映缓存分配器保留空间，都不等于整个设备上所有进程的占用。恢复进程的缓存和编译状态可能不同，显存与首次耗时也要分别解释。[6]

建议最终留下四组可对照日志：CPU 基线、CUDA FP32、CUDA BF16、BF16 加编译；再用相同有效批量比较累积与重计算。保存失败和退化的形状同样有价值。本实验提供一条可检查的计算链，让后续优化有共同对象；真实性能结论需要目标设备上的实际记录。

## 参考资料

[1] PyTorch，scaled_dot_product_attention：<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>。

[2] PyTorch，Automatic Mixed Precision：<https://docs.pytorch.org/docs/stable/amp.html>。

[3] PyTorch，torch.compile has different autograd semantics：<https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html>。

[4] PyTorch，torch.utils.checkpoint：<https://docs.pytorch.org/docs/stable/checkpoint.html>。

[5] PyTorch，Saving and Loading Models：<https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>；Reproducibility：<https://docs.pytorch.org/docs/stable/notes/randomness.html>。

[6] PyTorch，CUDA semantics：<https://docs.pytorch.org/docs/stable/notes/cuda.html>。资料核验：2026-10-25。
