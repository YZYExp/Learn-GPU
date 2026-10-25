# 第 19—24 章资料核验与实验记录

核验日期：2026-10-25。作者文件范围：chapters/19—24，examples/19、22、23、24 前缀。本记录属于编辑资料，不替代正文中的就近引用。

## 核实过的官方资料与原论文

| 资料 | 对应内容 | 当前注意事项 |
|---|---|---|
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 执行层级、异步、SIMT、协作和同步 | 当前新指南，页面标注更新于 2026-09-10。旧 `cuda-c-programming-guide` 已标为 Legacy，自 CUDA 13.0 后不再更新 |
| [CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html) | 原子作用域、数据竞争、消息传递 | 原子性和同步必须覆盖正确的参与者；原子操作不是任意数据的全局屏障 |
| [HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) | 线程协作、块、共享内存 | 不把所有 AMD Wavefront 一概写成 64 |
| [Porting CUDA code to HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) | HIPIFY、源码迁移与目标编译 | 源码转换不等于设备二进制兼容或性能等价 |
| [Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | 块级索引、掩码、包装 | 随书代码重新组织，未复制官方完整代码 |
| [Triton API](https://triton-lang.org/main/python-api/triton.language.html) | load/store/dot/reduction | 逻辑块大小与物理线程数分开解释 |
| [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | GEMM 分块、步长、尾部 | 教学内核未承诺厂商库性能 |
| [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | 行融合与片上复用 | 教学实现限定行宽与有限输入，不支持任意长行和全屏蔽行 |
| [Triton Gluon](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html) | 更显式控制布局的进阶方向 | 置于进阶位置，避免基础教学被最新语法淹没 |
| [PyTorch Event](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html) | 设备计时 | 分清设备区间、主机提交与端到端时间 |
| [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html) | 是否需要注册、Fake、别名与修改 | 能由普通组合表达时优先保留普通组合 |
| [Triton 与 torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) | triton_op 与结构化集成 | 直接调用与注册为算子的用途不同 |
| [Adding Training Registrations](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html) | register_autograd、上下文 | 不宣称一阶反向自动支持所有高阶能力 |
| [torch.library](https://docs.pytorch.org/docs/stable/library.html) | opcheck 与 gradcheck 的边界 | 契约检查与数学梯度检查不能互相替代 |
| [RMSNorm 原论文](https://arxiv.org/abs/1910.07467) | 均方根归一化定义 | 书中没有引用论文性能百分比作为现代 GPU 结论 |
| [CUTLASS](https://docs.nvidia.com/cutlass/) | 分块复用与输出阶段 | 不将 FP16 输入自动等同于特定矩阵指令执行 |

## 已完成的本地验证

原始环境具有 Python 与 NumPy；后续CPU隔离环境安装了PyTorch 2.14.1+cpu，仍没有GPU/Triton运行条件。

- `python3 examples/22-tiled-gemm-numpy.py`：10 组分块、非整齐尺寸和空维度组合通过，FP64 最大绝对误差约 6.22e-15。
- `python3 examples/23-reductions-numpy.py`：Softmax 平移不变性和行和检查通过；6 组 Softmax/RMSNorm 解析输入梯度与有限差分在所设容差下通过。最大绝对梯度差约 2.45e-5，出现在敏感的小维度样本；不将该差值推广为其他输入或低精度保证。
- `python3 examples/24-silu-gradient-numpy.py`：FP64 SiLU 有限差分检查通过，最大绝对梯度差约 1.17e-9。
- 7 份 Python 示例完成 `py_compile` 语法检查。

以下GPU脚本**尚未执行**：19-vector-add-triton.py、22-gemm-triton.py、23-row-softmax-triton.py。它们需要GPU、PyTorch、Triton和兼容工具链。24-custom-op-pytorch.py已在PyTorch 2.14.1+cpu实际运行，CPU契约、opcheck、gradcheck和编译前后向对照通过；其CUDA路径未执行。未给出GPU速度或加速比。

## 图片建议

实际图由主编统一生成并审校，本记录不放占位链接。

1. 第 19 章：1000 项划成 4×256 的队列，最后 24 项以灰色明确表示无效；如果使用生图，中文标签必须和正文核对。
2. 第 20 章：同一块的生产—屏障—消费—屏障与两个流的事件依赖分开画，不把块级屏障画成全设备屏障。
3. 第 21 章：逻辑数据块与物理线程映射分层示意，强调二者不是一一对应。
4. 第 22 章：固定输出块沿 K 移动，三幅连续图依次显示输入分块、部分和、完成输出，标注同一累加器身份。
5. 第 23 章：Softmax 的减最大值、指数、行和、归一化连续过程；同时对照 RMSNorm 的归约轴。
6. 第 24 章：前向保存的张量怎样被反向消费，以及 Fake 实现只传递元信息；不能用前向公式代替反向依赖。

## 交叉审稿后的接口修订

2026-10-25：落实 `editorial/review-scale.md` 的两项必要修正。

- 向量加法入口对非空输出比较底层 `untyped_storage().data_ptr()`，保守拒绝与任一输入共享存储；因此也拒绝同一存储内实际不重叠的切片。第19、21章同步写明限制，GPU脚本加入原地及偏移重叠切片的拒绝检查。空张量不启动内核，不以空存储的零地址判断别名。
- SiLU注册实验的真实与Fake路径调用同一纯元信息校验：仅连续strided、CPU/CUDA、FP32/FP64。两条路径拒绝整数/布尔/复数/未声明类型，避免实际类型提升与Fake的empty_like元信息不一致。第24章同步契约；脚本增加FP32检查及未支持类型的拒绝检查。
- GEMM GPU示例注明给定容差仅用于当前小K与标准正态样本，不推广为任意形状和输入尺度的保证。

修订后再次完成3份修改脚本语法检查，并重跑分块GEMM、Softmax/RMSNorm梯度、SiLU梯度的NumPy参考，结果与前述记录一致。后续PyTorch 2.14.1+cpu运行确认第24章的CPU契约拒绝、opcheck、gradcheck与编译检查通过；GPU拒绝检查仍未实际运行。详见 editorial/cpu-runtime-validation.md。

接口补充依据：[PyTorch torch.Storage](https://docs.pytorch.org/docs/stable/storage)说明普通视图共享底层存储，并可比较untyped storage的data_ptr；不将张量自身data_ptr与底层存储地址混用。
