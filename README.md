# GPU 与现代 AI 计算

> 从硬件、训练到算子优化 · 2026年10月版

一部面向有编程兴趣读者的中文图解知识书。以一次训练迭代为起点，连接硬件、自动求导、编译器、算子、性能与集群。

[在线阅读](https://yzyexp.github.io/Learn-GPU/) · [首版 EPUB](downloads/GPU-Modern-Computing.epub)

## 从这里阅读

[导读与阅读路线](appendices/reading-guide.md)

### 从一次训练看整个系统

- [第1章　一次训练迭代发生了什么](chapters/01-training-step.md)
- [第2章　张量、布局与矩阵运算](chapters/02-tensor-layout.md)
- [第3章　自动求导、反向传播与优化器](chapters/03-autograd-optimizer.md)
- [第4章　从 Python 到 GPU 执行的完整路径](chapters/04-execution-stack.md)

### GPU 内部与整机硬件

- [第5章　从图形处理到通用计算](chapters/05-gpu-evolution.md)
- [第6章　线程、Warp 与 SM：GPU 如何组织工作](chapters/06-execution-model.md)
- [第7章　寄存器、共享内存、缓存与 HBM](chapters/07-memory-hierarchy.md)
- [第8章　矩阵计算单元与数值格式](chapters/08-matrix-units.md)
- [第9章　封装、Chiplet、CPU 与 GPU 互连](chapters/09-system-interconnect.md)

### 把单卡训练做好

- [第10章　搭建可复现训练环境](chapters/10-reproducible-environment.md)
- [第11章　数据加载与计算流水线](chapters/11-input-pipeline.md)
- [第12章　显存账本与 OOM](chapters/12-memory-budget.md)
- [第13章　混合精度、FP8 与 FP4](chapters/13-low-precision.md)
- [第14章　重计算、梯度累积与训练稳定性](chapters/14-memory-stability.md)

### 从框架进入编译器

- [第15章　Eager、计算图与动态图](chapters/15-graphs-eager.md)
- [第16章　torch.compile、图断裂与动态形状](chapters/16-torch-compile.md)
- [第17章　算子融合、布局变换与代码生成](chapters/17-fusion-codegen.md)
- [第18章　JAX、XLA、MLIR 与编译路线](chapters/18-jax-xla-mlir.md)

### 亲手编写 GPU 算子

- [第19章　第一个 Kernel 与可靠计时](chapters/19-first-kernel.md)
- [第20章　CUDA 与 HIP 的执行和内存模型](chapters/20-cuda-hip.md)
- [第21章　Triton 的块级编程](chapters/21-triton.md)
- [第22章　GEMM：从朴素实现到分块计算](chapters/22-gemm.md)
- [第23章　Reduction、Softmax 与 RMSNorm](chapters/23-reductions.md)
- [第24章　融合算子、反向算子与框架集成](chapters/24-custom-autograd.md)

### 现代算子与性能工程

- [第25章　Roofline、Profiler 与瓶颈定位](chapters/25-roofline-profiler.md)
- [第26章　异步搬运、流水线与 Warp Specialization](chapters/26-async-pipelines.md)
- [第27章　Attention 与 FlashAttention](chapters/27-attention.md)
- [第28章　现代 Kernel DSL、自动调优与 AI 辅助优化](chapters/28-modern-dsl.md)

### 从单卡到训练集群

- [第29章　集合通信与网络拓扑](chapters/29-collectives.md)
- [第30章　DDP、ZeRO 与 FSDP](chapters/30-fsdp-zero.md)
- [第31章　张量、流水线与上下文并行](chapters/31-parallelism.md)
- [第32章　MoE 与专家并行](chapters/32-moe.md)
- [第33章　训练工程、检查点与故障恢复](chapters/33-reliability.md)

### 现代工作负载与生态选择

- [第34章　预训练、微调、蒸馏与强化学习后训练](chapters/34-training-workloads.md)
- [第35章　推理中的 Prefill、Decode 与 KV Cache](chapters/35-inference.md)
- [第36章　GPU 与专用加速器的系统比较](chapters/36-ecosystems.md)

## 附录

- [贯穿实验：训练一个小型因果 Transformer](appendices/transformer-lab.md)
- [实验索引与复现方法](appendices/lab-index.md)
- [术语：把计算层次分清](appendices/glossary.md)

## 构建与复现

```bash
python -m pip install -r requirements-book.txt
python scripts/build.py --check
python scripts/build.py
```

输出为 `build/GPU-Modern-Computing.epub` 和合并 Markdown。EPUB包含分章导航、中文图注、数学公式图像和全部正文插图；不生成网站。章节与资源路径已为将来主页发布准备，未来可直接消费 `book.json`。

`examples/` 包含20份实验脚本。标准库与NumPy实验可直接运行；PyTorch/JAX/Triton按各自官方环境说明安装。GPU、AMD、Apple与多卡支持取决于具体设备、系统、框架与工具版本。实验执行边界见[实验索引](appendices/lab-index.md)与[验证记录](editorial/verification.md)。

## 主页与 GitHub Pages

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --base-path /Learn-GPU/ --epub downloads/GPU-Modern-Computing.epub
```

网页只需Markdown、Pillow和latex2mathml；数学以MathML离线生成。支持手机目录、全文搜索、字号调整与分章导航。Pages和整书导出为独立工作流，首次发布需将Pages Source设为GitHub Actions。

## 配图与资料

全书36张正文图，加封面1张；其中12张为生成模型制作的教学示意（含封面），25张为脚本绘制的精确图解。全部正文图配中文解释；概念插画不代表真实型号内部结构或测量结果。

生成图的用途、提示词与修订记录在 `editorial/imagegen-prompts.json`，逐图索引在 `editorial/image-index.json`，均作为编辑资料，不进入正文。精确图可用三个 `scripts/figures-*.py` 重建。字体来自Noto CJK项目，附SIL Open Font License于 `assets/fonts/OFL.txt`。

每章有就近引用和本章资料；当代技术的资料核验日期为2026-10-25。版本与支持矩阵会变化，不将上游性能当成本书实测。

## 项目结构

- `chapters/`：36章中文Markdown主稿
- `appendices/`：导读、贯穿实验、实验索引、术语
- `assets/images/`：实际入书的37张图片
- `assets/fonts/`：重绘用中文字体与许可
- `examples/`：可运行实验与教学内核
- `references/`：分组调研记录与来源
- `editorial/`：交叉审稿、图索引和验证记录
- `scripts/`：校验、成书、精确图重建
- `book.json`：章节/附录顺序与稳定路径
- `build/`：可再生成产物，不进入Git

## 版本历史

根提交只含零字节README，随后逐步叠加目录、正文、实验、配图与审稿修订。工程包附 `book-history.bundle`；需要还原Git历史时运行：

```bash
git clone book-history.bundle gpu-book-history
```

仓库：[YZYExp/Learn-GPU](https://github.com/YZYExp/Learn-GPU)。网页源代码与轻量Pages CI已纳入工程；发布说明见 [site/README.md](site/README.md)。
