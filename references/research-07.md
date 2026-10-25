# 第7—12章调研与验证记录

核验日期：2026-10-25。正文为独立解释和明确假设的教学推导；没有将厂商峰值或第三方性能作为本书实测。

## 官方来源

- CUDA新编程指南（已打开核验）：https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html 。芯片层级、共享内存、L1及线程块集群。
- CUDA Best Practices（已打开）：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html 。合并访问、共享内存、寄存器和优化方法。
- CUTLASS Overview（已打开）：https://docs.nvidia.com/cutlass/latest/overview.html 。2026-10版本4.8.1，列出初始Rubin支持。页面同时残留CuTe DSL公测结束预测，正文没有将该旧状态当当前结论。
- PyTorch Numerical Accuracy：https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html 。TF32及误差。正文避免固定易变默认值。
- Transformer Engine FP8/FP4 primer：https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html 。低精度与缩放，具体训练配方留第13章。
- AMD CDNA：https://www.amd.com/en/technologies/cdna.html 。Chiplet、HBM及Infinity结构。
- AMD CDNA4白皮书：https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf 。封装结构。
- NVIDIA Blackwell官方技术概览：https://resources.nvidia.com/en-us-blackwell-architecture 。数据中心双裸片。
- NVLink-C2C：https://www.nvidia.com/en-us/data-center/nvlink-c2c/ 。Grace Hopper/Blackwell一致性互连。
- Grace调优指南：https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html 。NUMA与CPU/GPU位置差异。
- CUDA Multi-GPU（已通过指南目录点击确认真实路径）：https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html 。不可用猜测的programming-systems-with-multiple-gpus.html。
- PyTorch Reproducibility（已打开版本化页面）：https://docs.pytorch.org/docs/2.14/notes/randomness.html 。随机性及确定性范围。
- PyTorch HIP semantics（已打开）：https://docs.pytorch.org/docs/2.14/notes/hip.html 。ROCm沿用torch.cuda命名。
- PyTorch DataLoader：https://docs.pytorch.org/docs/stable/data.html 。worker、prefetch、pinning。
- Pinmem/nonblocking教程：https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html 。页锁定、异步、重叠的条件区别。
- 数据加载优化教程：https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html 。预取与全流程测量。
- PyTorch CUDA Memory（已打开）：https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html 。OOM快照与框架外分配不可见。
- PyTorch CUDA semantics：https://docs.pytorch.org/docs/stable/notes/cuda.html 。allocated/reserved及empty_cache。
- FlashAttention原论文：https://arxiv.org/abs/2205.14135 。避免物化注意力矩阵。

## 建议图

1. 第7章：生成一张半剖面教学插画，从GPU模块逐层靠近HBM、L2、SM共享内存和寄存器；中文标注，明确为概念图，比例不表示真实尺寸。精确地址映射另用确定性图。
2. 第8章：生成矩阵单元与数据块的近景示意，不能把Tensor Core画成拥有全部矩阵的单线程；数值格式位宽表和运算计数用正文。
3. 第9章：从封装、GPU模块、服务器到机架的尺度递进插画，几张独立竖向图优于横排六格。区分封装内与设备间互连。
4. 第11章：可生成机房/工作站实景辅助；准备30ms、复制8ms、计算20ms的精确流水线时间图应程序绘制。
5. 第12章：训练显存账本适合精确堆叠图，16字节/参数只适用于本例假设，不能画成普遍公式。

## 已执行验证

- examples/10-environment.py：标准库入口成功，报告基础Python3.12.14且未安装PyTorch。后续CPU隔离环境已安装PyTorch 2.14.1+cpu并运行CPU训练，详见 editorial/cpu-runtime-validation.md；仍没有运行GPU训练。
- examples/12-memory-budget.py：默认输出16 GB / 14.901161 GiB；排除激活、工作区与框架外开销。
- 教学计算已核对：32×32输出块、K段32，朴素65536元素请求/理想2048元素搬入；FP16双子块4096字节，双缓冲8192字节。
- 4096立方GEMM为137438953472 FLOP；4×2048×2048×2字节激活为32 MiB；4×16×2048²×2注意力矩阵为512 MiB。
- 1 GiB/25 GB/s约42.95ms；各章严格区分GB与GiB。

## 未执行限制

无GPU、无PyTorch，章内PyTorch样例只经过人工核对，不宣称功能、收敛或性能已测。版本/安装环境必须由读者或后续具设备环境重新确认。
