# 第13—18章核验与配图建议

核验日期：2026-10-25。以下章节已写完整正文，非提纲。无实测GPU性能数字。正文中的24N/8N访存、低精度载荷、编译回本步数均为明确假设下的教学计算。

## 官方资料与使用范围

- PyTorch AMP及AMP examples：autocast与梯度缩放职责不同，FP16累积时unscale与clip顺序。https://docs.pytorch.org/docs/stable/amp.html ；https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Transformer Engine FP8 primer与支持矩阵：FP8配方、块缩放、MXFP8/NVFP4和具体计算能力支持边界。https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html ；https://nvidia.github.io/TransformerEngine/support_matrix.html
- NVFP4原论文：配方验证不能推广为任意程序只换dtype。https://arxiv.org/abs/2509.25149
- PyTorch checkpoint：显式use_reentrant、非重入实现、RNG与重算。https://docs.pytorch.org/docs/stable/checkpoint.html
- PyTorch CUDA semantics、Autograd mechanics、FX文档：异步执行、求导图、FX实验与Dynamo区别。https://docs.pytorch.org/docs/stable/notes/cuda.html ；https://docs.pytorch.org/docs/stable/notes/autograd.html ；https://docs.pytorch.org/docs/stable/fx.html
- PyTorch2原论文：图捕获/代码生成职责。https://pytorch.org/assets/pytorch2-2.pdf
- 当前PyTorch编译文档：graph_breaks、fullgraph、dynamic、recompile。https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html ；https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html ；https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html ；https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html
- CUTLASS、XLA GPU架构：epilogue、布局选择与融合。https://docs.nvidia.com/cutlass/latest/ ；https://openxla.org/xla/gpu_architecture
- JAX thinking/JIT/async/Pallas：函数变换、抽象追踪、同步计时与内核层。https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html ；https://docs.jax.dev/en/latest/jit-compilation.html ；https://docs.jax.dev/en/latest/async_dispatch.html ；https://docs.jax.dev/en/latest/pallas/
- XLA、StableHLO、MLIR各自角色：https://openxla.org/xla ；https://openxla.org/stablehlo ；https://mlir.llvm.org/

## 建议加入的模型生图（由主编整合）

1. 第13章：同一组数值对应整张量共享缩放和小块独立缩放的对照，红色异常值只影响局部。宜用生图模型生成清晰的中文教学信息图，再人工核对数值/箭头。不要用生图假装展示真实位编码或实测误差。
2. 第14章：保存全部沿途材料与只保存边界、需要时重走的空间对照。生图可用整洁的半写实计算车间场景，比喻明确标注；精确张量生命周期另用矢量图。
3. 第15—18章：从Python工作台、图结构规划桌到GPU执行车间的连续纵向剖面，区分软件层与硬件层。仅作为AI教学示意，不用它证明编译器真实内部结构。
4. 第17章：分开三个工序反复走仓库、融合为一个连续工位的对照。配文明确这是数据流比喻，真实字节核算在正文。

## 示例与验证

- examples/14-accumulation.py：PyTorch CPU双精度，等大小微批次累计梯度和全批平均梯度对照。无BatchNorm/Dropout。
- examples/15-fx-graph.py：PyTorch CPU，FX symbolic_trace观察，不冒充Dynamo路径。
- examples/16-compile-shapes.py：PyTorch CPU/默认编译后端，32/48/32形状校验；依赖正常编译工具链，不预言日志次数。
- examples/18-jax-boundaries.py：JAX CPU或可用设备，首次与后续调用同步计时，打印jaxpr。
- 基础Python仍不含torch/jax；后续CPU隔离环境安装PyTorch 2.14.1+cpu，14/15/16脚本实际运行通过，其中16包含CPU Inductor编译和32/48/32形状。18的JAX脚本及所有GPU路径仍未执行。详见 editorial/cpu-runtime-validation.md。
