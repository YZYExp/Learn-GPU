# 第25—30章资料、教学图与核查记录

核验日期：2026-10-25。正文采用章内编号引用。仅依据原论文、作者发布材料与官方文档；没有引用第三方跑分，也没有把未实测内容写成实测。

## 已完成正文

- `chapters/25-roofline-profiler.md`：时间线到内核指标、Roofline、口径、占用率、MFU、有效token；计算示例明确是假设。
- `chapters/26-async-pipelines.md`：生产者消费者、双缓冲代次、TMA、Warp分工、资源代价与边界；异步流程标为教学伪代码。
- `chapters/27-attention.md`：Attention形状、在线Softmax、重计算、FA1/2/3/4演进、实际后端、两块手算与稠密复杂度。
- `chapters/28-modern-dsl.md`：Triton/CuTe DSL/cuTile/Helion/Pallas/TileLang的比较；调优摊销、独立评估、AI内核验证与维护成本。
- `chapters/29-collectives.md`：语义、拓扑、环形量级估算、重叠、强弱扩展、Rank延迟排查与测量口径。
- `chapters/30-fsdp-zero.md`：DDP/ZeRO stages/FSDP2、16字节假设账本、AllGather生命周期、峰值显存与预取、检查点与累积。

## 核查的关键事实与来源

1. Nsight Compute支持Roofline与分层Roofline；计数器采集会影响运行，必须区分应用运行与采集。https://docs.nvidia.com/nsight-compute/ProfilingGuide/
2. CUTLASS现代GEMM文档描述Hopper TMA与生产者消费者Warp分工；Blackwell CuTe DSL样例有对应架构约束。https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html ；https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm
3. FlashAttention原论文提出IO-aware exact attention；FA2改进并行与工作划分；FA3针对Hopper异步低精度。https://arxiv.org/abs/2205.14135 ；https://arxiv.org/abs/2307.08691 ；https://arxiv.org/abs/2407.08608
4. FlashAttention-4的2026作者博客确实存在，面向Blackwell，强调矩阵算力与其他资源增长不对称，CuTe DSL实现。正文只解释核心方向，不复述未经本工程重现的性能数字。https://tridao.me/blog/2026/flash4/
5. cuTile是CUDA Tile路线，不是CuTe DSL的另一拼法。https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html
6. Helion具有调优和缓存部署机制。https://helionlang.com/deployment_autotuning.html
7. Pallas是JAX GPU/TPU内核路线；后端/API较快变化，未做全面支持承诺。https://docs.jax.dev/en/latest/pallas/
8. TileLang官方文档与仓库已核验，只介绍路线，不以版本号作优劣结论。https://tilelang.com/ ；https://github.com/tile-ai/tilelang
9. NCCL集合通信要求各Rank匹配参与；RCCL支持AMD多GPU多节点通信。https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html ；https://rocm.docs.amd.com/projects/rccl/en/latest/
10. DDP不自动切分输入；ZeRO stage 1/2/3分别扩展分片范围；PyTorch当前FSDP2使用fully_shard，通信边界由模块分片组织，状态使用DTensor。https://docs.pytorch.org/docs/stable/notes/ddp.html ；https://www.deepspeed.ai/tutorials/zero/ ；https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html

## 建议的实际教学图（供主编生成并整合）

- 第25章：一个人查看GPU系统时间线的场景生图，显示“搬运、计算、等待”三种活动；精确Roofline数值曲线必须另用绘图工具。不能让生图虚构可读跑分。
- 第26章：生产者搬运与消费者计算的工位隐喻，两个缓冲台轮换；图注明确为协作机制隐喻。精确时间与屏障关系另以矢量图呈现。
- 第27章：Attention从整张巨大分数表转换为分块逐次处理的教学插画，中文标签“查询、键和值、局部计算、输出”；具体Softmax合并用正文手算核对。
- 第28章：从数学公式到高层分块、再到底层硬件的分层工作台，避免把所有DSL画成单一线性流水线。
- 第29章：同机GPU与跨机网卡、交换设备的场景，突出近远层次；不是特定真实机器的布线复刻。
- 第30章：DDP各卡保留完整副本与FSDP各卡保存参数分片、需要时组装的对照示意。各状态块的精确身份与容量需程序绘图或主编核验。

## 可运行内容与限制

- `examples/25-roofline_estimator.py`：纯Python、无依赖，验证第25章理论Roofline与Amdahl数值。已运行通过；没有GPU计时。
- `examples/27-online_softmax.py`：纯Python在线Softmax加权和，含独立两元素手算以及随机长度/分块/极端有限分数的前向对照。已运行通过，最大绝对误差约1.11e-16。仅CPU数学验证，不含GPU并行、反向、Dropout与全屏蔽行语义。
- 本分工没有NVIDIA/AMD多卡设备实测；TMA、FA家族、DSL、NCCL/RCCL、FSDP2的性能与环境组合未在本工程复现。
