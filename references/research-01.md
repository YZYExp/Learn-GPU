# 第 1—6 章调研与图文整合建议

核验日期：2026-10-25。内容负责人：foundations。

## 已写章节

- 01-training-step.md：以单标量模型连接 Transformer 批次、前向、反向、更新与时间线；区分 batch、step、epoch。
- 02-tensor-layout.md：形状、步幅、视图、广播、逻辑字节、布局契约；含可手算的转置。
- 03-autograd-optimizer.md：链式法则、矩阵反向、动态求导图、模式、AdamW、中心差分。
- 04-execution-stack.md：模块/算子/kernel/指令，分派、库、编译、驱动、PTX/cubin、异步流。
- 05-gpu-evolution.md：以解决的问题组织历史，避免产品年表；架构/板卡/系统命名区分。
- 06-execution-model.md：网格、块、线程、SM、warp、分歧、驻留、占用率，AMD CDNA/RDNA差异。

## 官方资料与核查重点

1. PyTorch Autograd mechanics：前向时建立求导依赖，保存反向需要的张量，mode与eval分离。https://docs.pytorch.org/docs/stable/notes/autograd.html
2. PyTorch Training/Optimization Tutorials：梯度清理、反向、optimizer step。https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
3. Tensor Views与Tensor.view：转置视图不必复制，reshape可能复制，stride兼容条件。https://docs.pytorch.org/docs/stable/tensor_view.html ；https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
4. 2026 PyTorch DevLog Dispatcher：接口到后端的分派路径。https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/
5. CUDA Programming Guide CUDA platform与Asynchronous Execution：PTX/cubin/driver阶段，流内顺序与主机异步。https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html ；https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
6. CUDA Advanced Kernel Programming：SM与warp，Volta独立线程调度不会免除同步。https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
7. AMD HIP Hardware features + AMD ISA Blog：CDNA wave64、RDNA wave32/64支持按架构与编译模式查询。https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html ；https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
8. NVIDIA CUDA origins、Blackwell Tuning Guide：CUDA公布时间与现代架构能力按具体型号核对。https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/ ；https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html

正文邻近标注 [n]，章末给出独立参考。稳定原理没有引用第三方博客。未声明任何GPU性能实测结果。

## 图片建议（尚未写入图片链接，等待主编实际资产）

| 章节 | 适合生图的画面 | 精确图另绘的关系 |
|---|---|---|
| 1 | 一张桌面训练机到机房的清晰场景，模型输入和输出可视化 | 前向→损失→反向→更新四阶段，参数/激活/梯度分色 |
| 2 | 数据块的实体化教学场景，可作章引 | 存储0—5，同一数组对应原矩阵/转置/reshape三种解释，标签中文 |
| 3 | 计算网络向后追踪影响的教学情景 | x,w→z→L数值和两段导数，前后向箭头分开 |
| 4 | CPU主机、GPU板卡与屏幕上的代码的整机场景 | Python→分派/图编译→库或生成kernel→runtime/driver→GPU；异步双时间线 |
| 5 | 从图形工作站到AI机房的技术演进插画，不冒充历史实物 | 固定功能→可编程→通用→矩阵路径的因果关系，不按真实芯片比例 |
| 6 | GPU内部多工作组的概念教学插画，避免画成真实电路剖面 | 网格4块，每块256线程，末块24无效；warp32分组；AMD差异放图旁 |

精确关系应由可校验的代码/矢量制作，不能以生图中的箭头或芯片细节作证。所有生图注明AI教学示意。

## 实验与验证

- `examples/01-training-cpu.py`：标准库运行，解析梯度与中心差分相符，参数更新后损失为0（特制单样本）。已实际执行通过。
- `examples/02-layout-cpu.py`：标准库运行，同存储的转置与reshape数值正确。已实际执行通过。
- 正文Python代码块已通过AST语法检查。
- 后续建立CPU隔离环境，使用PyTorch 2.14.1+cpu实际运行第1—3章的全部内嵌Python片段，通过；GPU/driver/stream例子仍为解释。完整记录见 editorial/cpu-runtime-validation.md。
