# 第31—36章资料、图解与验证记录

核验日期：2026-10-25。作者任务范围：chapter 31—36；未修改共享目录与Git。

## 资料策略

当前实现以官方文档与项目仓库为准，原理补充原论文。完整来源逐章放在正文末尾，正文[1]等就近引用。已联网核查：

- NVIDIA Megatron Core parallelism guide、context parallel API、MoE token dispatcher、MoE 0.17 guide，明确张量/流水线/上下文/专家并行区别。
- PyTorch Distributed Checkpoint 2.14文档、async checkpoint教程；保存布局/版本兼容限制与异步暂存成本。
- vLLM官方博客、官方分页缓存设计、v0.20分离prefill文档；该分离路径按实验性描述。
- TRL Distillation/SFT/DPO/GRPO文档、PEFT量化、verl官方文档；区分离线数据与在线生成、训练/推理布局、策略滞后。
- MLX统一内存/惰性求值、Intel原生XPU入门、Google TPU架构、ROCm Windows硬件/组件矩阵。没有将Windows/WSL/Linux视为相同覆盖。
- Megatron 2021论文、Switch Transformers、LoRA、PagedAttention与推测解码原论文用于机制；没有采用旧硬件实测结果代表现代性能。

## 原创图解建议（主编整合）

| 图号建议 | 章节 | 生图教学目的 | 准确性约束 |
|---|---|---|---|
| F31 | 31 | 四张GPU上的模型分工：同样四卡分别切层、切矩阵、切序列的视觉对照 | 图内需中文；不是物理芯片结构；精确分片建议配矢量核对 |
| F32 | 32 | 四卡、八专家的token派发与返回场景，显示一张卡拥挤、另一张较空 | 不把专家画成已知固定语言人格；dispatch和combine双向 |
| F33 | 33 | 训练前台、主机暂存、后台存储三处的检查点场景 | 异步仍消耗内存与带宽；未完成写入不得标安全 |
| F34 | 34 | 同一个模型的SFT与RL工作空间：数据→更新，对照生成→环境→更新 | 提示词和回答区分，工具等待属于流水线 |
| F35 | 35 | 提示词整段进入、回答逐token出现与缓存增长的场景 | KV为历史状态不是回答文本；精确容量另用公式 |
| F36 | 36 | 台式统一内存设备、GPU服务器与专用加速器机架的多尺度场景 | 不虚构真实品牌板卡、封装或厂商参数 |

## 本地验证

- `python3 examples/33-resume-demo.py`：标准库CPU程序实际运行，完整恢复与连续四十步状态一致；故意删除动量后产生不同参数。
- `python3 examples/35-kv-budget.py`：默认32层、8 KV头、128头维度、BF16、8192长度输出每token128KiB，总1GiB。
- `python3 examples/35-kv-budget.py --batch 16`：总16GiB。
- 以上均为教学演示；没有运行GPU、分布式训练、MoE或推理服务器性能测量。

## 编辑注意

章节31—36完整正文已落盘，约每章1800—2100个汉字，表格/公式/引用另计。没有插入未生成图片的占位引用；主编可在每章引言之后添加实际图片，保持相对路径与中文图注。
