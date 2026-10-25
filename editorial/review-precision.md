# 第1—12章交叉审稿

审稿范围：事实、公式数值、现代硬件支持与示例正确性。只读章节，未修改章文件。核验：2026-10-25。

## 必要修正

### 1. 第1—3章行内数学定界符缺损（高优先级）

- `chapters/01-training-step.md:11`：`(8\times256\times512=1,048,576)` 等实际只有普通圆括号，会暴露LaTeX命令。
- `chapters/01-training-step.md:44`：`(w\leftarrow w-\eta g\)` 缺行内数学起始定界符，并有不成对结束符。
- `chapters/02-tensor-layout.md:9`、`:11`：矩阵维度、集合符号和FLOPs公式也只有普通圆括号。
- `chapters/03-autograd-optimizer.md:7`、`:23`：`(z=xw\)`、`(Y=XW\)` 等同样损坏。

建议统一改用构建工具确实支持的 `$...$` 或 `\(...\)` 成对定界，并在最终EPUB/阅读成品检查，不是只让源文件看起来像公式。变量名的普通圆括号可保留；含命令的公式必须处理。

### 2. 第3章有限差分公式包含实际退格控制字符（高优先级）

`chapters/03-autograd-optimizer.md:75` 中原本应为 `\bigl`、`\bigr` 的位置，实际出现两个 0x08 字节，后面分别跟 `igl` 与 `igr`。很可能是写文件时Python非raw字符串把 `\b` 转义。

建议直接重写该公式为 `$[L(w+h)-L(w-h)]/(2h)$`，并扫描全书不允许的控制字符。这不是审美意见，当前字节已不是有效公式。

### 3. 第2章的起始偏移练习会越界（中优先级）

`chapters/02-tensor-layout.md:65` 让读者把 `examples/02-layout-cpu.py` 的起始偏移增加一位。脚本底层存储是 `list(range(6))`，原2×3形状与步幅(3,1)最大索引为5；offset=1后最大索引为6，必然越界。

已在只读导入脚本后复现 `IndexError`。建议把练习改为“先将底层列表扩为7个元素，再把起始偏移设为1”，或使用原存储内更小的切片形状。原脚本默认运行本身正确。

## 已重点核验、无需修正

- 第1章手算梯度(-12,-6)、更新后预测5；第2章2BSDF约4.29e9 FLOPs；第7章块复用65536/2048=32及双缓冲8192字节；第8章4096³ GEMM约1374亿FLOPs；第9章1GiB/25GBps约43ms；第11章1300/3000有效token约43%；第12章16GB状态、32MiB激活、512MiB注意力矩阵等教学数值核算一致，且正文有条件说明。
- 第6章明确区分CDNA wave64与RDNA wave32/可选wave64，没有将AMD固定写为64；没有发现需要更正的通用分组或同步结论。
- 第8章提及Rubin初始支持确有官方依据，不应当因为新而删除。CUTLASS官方overview明确列出Rubin SM107初始支持、CuTe相关构建块以及预览性质；正文已区分接口加入与应用稳定收益。官方地址：https://docs.nvidia.com/cutlass/latest/overview.html 。核查位置包括“Initial Rubin support”“Dense and blockscaled GEMMs ... preliminary Rubin support”“Added initial Rubin support (SM107)”。建议最终来源档案保留核验日期，避免latest变化后失去上下文；无需为本章增加未经实测的型号性能数字。
- 第5章CUDA于2006年公布的说法有NVIDIA官方企业时间线和CUDA编程指南支持，与2007年某些版本/工具发布节点不矛盾。
- 第10—11章内嵌PyTorch代码只做功能示范，未发现明显语法或逻辑错误；GPU/ROCm实际执行仍需对应环境验证。
