# 理解 GPTQ 算法机制

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/gptq-mechanics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解基础训练后量化 (quantization) (PTQ) 的局限性是必要的。像将权重 (weight)四舍五入到最近整数值这样的简单方法，通常无法保持模型精度，尤其是在 INT4 等极低位宽下。大型语言模型对此特别敏感，因为在一个层中的小错误可能会在通过后续层传播时累积放大。

GPTQ (广义训练后量化) 的发展旨在解决这种精度损失问题。GPTQ 并非独立量化每个权重，而是同时对整个层的权重矩阵进行优化量化过程，以尽量减少该层输出中引入的误差。这是一种更精密的方案，比基础 PTQ 方法能获得显著更好的精度，通常可媲美原始 FP32 模型的性能，且无需重新训练。

### 逐层量化 (quantization)问题

GPTQ 每次操作一个层。考虑 LLM 中的一个线性层，其运算由矩阵乘法定义：


$$
Y = WX
$$


这里，$W$ 是原始全精度权重 (weight)矩阵，$X$ 是输入激活（来自前一层或初始嵌入 (embedding)），而 $Y$ 是输出激活。当我们把权重量化到较低精度，例如 INT4 时，得到量化权重矩阵 $WQ$。PTQ 的目标是找到一个 $WQ$，使得输出 $WQ X$ 尽可能接近原始输出 $WX$。

该层中由量化引入的误差可以通过原始输出与量化输出之间的均方误差 (MSE) 来衡量：


$$
\text{误差} = ||WX - WQ X||^2_F
$$


其中 $|| \cdot ||^2_F$ 表示平方 Frobenius 范数（所有元素平方差之和）。基础 PTQ 方法，例如四舍五入 (RTN)，独立确定 $WQ$ 中的每个元素，通常仅仅通过对 $W$ 中对应元素进行四舍五入。这种贪婪方法在最小化整体输出误差时，没有考虑输入数据 $X$ 的结构或权重之间的互相影响。

### 序列量化 (quantization)与误差补偿

GPTQ 采用一种更细致的迭代方法。它按序列量化层内的权重 (weight)，通常是逐列（有时是逐行，或以小块形式）。核心思想是**误差补偿**：在量化特定权重（或一组权重）后，GPTQ 会计算此量化步骤引入的误差，并立即更新层中*剩余尚未量化*的权重，以抵消该误差。

设想逐一量化权重矩阵 $W$ 的列。

1. 选择第一列 $w_1$。
2. 找到此列的最佳量化值 $wq_1$。这不一定仅仅是四舍五入；选择它是为了最小化重建误差 $||w_1 x_1 - wq_1 x_1||^2$，可能需要考虑其对层其余部分的影响。
3. 计算此列的量化误差：$E_1 = (w_1 - wq_1) X$。
4. **关键步骤：** 调整*剩余*未量化的列 ($w_2, w_3, ...$) 以补偿 $E_1$。目的是修改这些列，使其组合输出在乘以 $X$ 时，有助于抵消误差 $E_1$。
5. 移至下一列 $w_2$，对其进行量化，计算误差 $E_2$，并更新剩余列 ($w_3, w_4, ...$)，依此类推。

这种序列过程确保早期做出的量化决定在后期得到补偿，从而使该层的整体重建误差远低于独立四舍五入方法。

### 二阶信息 (Hessian) 的作用

剩余权重 (weight)究竟如何更新？简单地按比例加回误差可能并非最佳方案。GPTQ 使用关于误差函数的近似二阶信息，具体来说是运用 Hessian 矩阵。

The objective is to minimize the squared error $|| (W - WQ) X ||^2$. The Hessian of this objective function with respect to the weights $W$ provides information about the curvature of the error surface. For a linear layer, this Hessian $H$ is directly related to the input activations $X$:


$$
H = 2 X X^T
$$


该矩阵 $X X^T$ 代表输入特征的协方差。从直观上说，它根据校准数据，告诉我们输入空间中哪些方向最为重要或变化最大。

GPTQ 使用此 Hessian 矩阵的逆矩阵 $H^{-1}$ 来指导误差补偿步骤。当量化 (quantization)权重引入误差 $E$ 时，应用于剩余权重的更新与 $H^{-1} E$ 成比例。使用逆 Hessian 有助于根据输入数据统计信息，优先调整最敏感的方向。它允许进行更具针对性的补偿，将调整集中在对减少最终输出误差影响最大的位置。

计算和求逆完整的 Hessian 矩阵可能计算成本高昂，特别是对于大型层。GPTQ 采用高效的数值方法和近似法（通常按块操作或使用迭代求解器，如 Cholesky 分解）来处理此问题，使过程变得可行。此方案受到先前工作（如 Optimal Brain Surgeon 和 Optimal Brain Quantizer (OBQ)）的启发，这些工作也使用 Hessian 信息进行模型剪枝和量化。

### GPTQ 算法概述

以下是针对单个层权重 (weight)矩阵 $W$ 的 GPTQ 过程概述：

1. **初始化：** 从全精度权重 $W$ 开始。获取一个有代表性的校准数据集 $X$。
2. **计算 Hessian：** 使用校准数据计算 Hessian $H = 2 X X^T$ 或其近似值。计算其逆矩阵 $H^{-1}$（或准备好使用它进行高效更新）。
3. **初始化量化 (quantization)权重：** 设置 $WQ = 0$ 或某个初始猜测值。记录哪些权重已量化。
4. **迭代量化：**
   - 对于 $W$ 中的每个权重（或列，或块）$w_i$：
     - 如果 $w_i$ 已量化，则继续。
     - 确定 $w_i$ 的最佳量化值 $wq_i$。这涉及考虑当前已累积的量化误差及其对层输出的影响，通常是最小化 $|| (w_i - wq_i) X + \text{累积误差} ||^2$。
     - 计算此步骤引入的量化误差：$\Delta_i = (w_i - wq_i)$。
     - 更新*剩余未量化*的权重 $w_j$（其中 $j > i$）以补偿此误差。更新规则涉及 $\Delta_i$ 和逆 Hessian $H^{-1}$。例如，对 $w_j$ 的更新可以根据从 $H^{-1}$ 推导出的相应行/列互相影响来计算。
     - 将 $w_i$ 标记 (token)为已量化 ($WQ_i = wq_i$)。存储误差贡献或更新累积误差。
5. **最终确定：** 所有权重处理完毕后，$WQ$ 就是该层的最终量化权重矩阵。

> 单个层内 GPTQ 算法的简化流程。它迭代地量化权重，计算误差，并使用 Hessian 信息通过更新剩余权重来进行补偿。

### 校准数据与粒度

与基础 PTQ 类似，GPTQ 需要一小部分校准数据集（输入激活 $X$ 的样本）来计算 Hessian $H = 2 X X^T$。此数据集的质量和代表性会影响最终量化 (quantization)精度。通常，只需几百到一千个样本。

GPTQ 通常以**分组粒度**应用。权重 (weight)不是单独量化或逐列量化，而是分块处理（例如，每组 128 个）。这降低了 Hessian 计算和更新的计算开销，同时仍提供比逐张量或逐通道量化显著更好的结果。误差补偿机制应用于这些分组。

通过考虑层的重建误差并运用输入统计信息（通过 Hessian）来指导误差补偿，GPTQ 比简单方法更有效地最小化精度损失，使其成为在 LLM 中实现准确低位权重量化的常见方案。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.04615) — Elias Frantar, Saleh Ashkboos, Maximilian Seitzer, and Dan Alistarh (2022)
  Journal: arXiv preprint; Publisher: arXiv; DOI: [10.48550/arXiv.2210.04615](https://doi.org/10.48550/arXiv.2210.04615)
  本文介绍了GPTQ算法，阐述了其针对大型语言模型量化的逐层、序列化和基于Hessian误差补偿的方法。

---

[上一节](01-GPTQ%E4%BB%8B%E7%BB%8D.md) · [下一节](03-AWQ%EF%BC%9A%E6%BF%80%E6%B4%BB%E6%84%9F%E7%9F%A5%E6%9D%83%E9%87%8D%E9%87%8F%E5%8C%96.md)
