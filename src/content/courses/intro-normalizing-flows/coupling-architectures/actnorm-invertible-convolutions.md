---
course: "intro-normalizing-flows"
chapter: "coupling-architectures"
lesson: "actnorm-invertible-convolutions"
sourceId: 8271
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-4-coupling-architectures/actnorm-invertible-convolutions"
title: "ActNorm 与可逆卷积"
description: "了解 Glow 模型中的创新点，包括激活归一化和 1x1 卷积。"
order: 4
plots: []
sourceHash: "0bebbe5f238916267cfe304b66f83dd2de74b188c286b9b312dc2c78458ca070"
sourceCorrections: []
---

Glow 架构为耦合网络引入了两项结构性改进：激活归一化 (normalization)（Activation Normalization，通常简称为 ActNorm）和可逆 $1 \times 1$ 卷积。这些组件通过取代标准的批归一化（Batch Normalization）和固定的通道置换，提高了流模型在训练时的稳定性和性能。

用于高分辨率图像的归一化流需要消耗大量内存。这往往迫使研究者使用非常小的批次大小（Batch Size），有时甚至每个批次仅包含一张图像。由于批次统计量会变得极具噪声，标准批归一化在这种场景下表现不佳。ActNorm 通过应用仿射变换来解决这一限制，其缩放和平移参数 (parameter)（scale and bias）根据第一批数据进行初始化。

对于具有空间维度和通道的输入张量 $x$，ActNorm 应用通道级别的缩放 $s$ 和偏置 (bias) $b$：

$y_{i, j, c} = x_{i, j, c} \cdot s_c + b_c$

这里，$i$ 和 $j$ 表示空间坐标，$c$ 表示通道索引。在第一次前向传播过程中，$s$ 和 $b$ 被初始化，使得给定批次的输出激活值均值为零，方差为一。在初始步骤之后，$s$ 和 $b$ 被视为与批次统计量无关的常规可训练参数。

由于这是一个仿射变换，其雅可比行列式的对数计算非常直接。它仅是缩放参数绝对值的对数之和乘以空间维度：

$\log \det J = h \cdot w \cdot \sum_{c} \log |s_c|$

其中 $h$ 和 $w$ 分别是输入张量的高和宽。

在 RealNVP 中，各部分数据之间的信息混合是使用固定操作（如反转通道顺序）完成的。虽然这种方法有效，但固定置换限制了模型的灵活性。$1 \times 1$ 卷积是应用于通道维度的线性变换。通过将固定置换替换为可学习的 $1 \times 1$ 卷积，模型可以自动学习最有效的通道融合方式。

设 $x$ 为形状为 $(h, w, c)$ 的张量，$W$ 为 $c \times c$ 的权重 (weight)矩阵。每个空间位置的卷积操作为：

$y_{i,j} = W x_{i,j}$

为了符合归一化流的要求，$W$ 必须是可逆的。该操作在整个空间网格上的雅可比行列式对数为：

$\log \det J = h \cdot w \cdot \log |\det(W)|$

计算 $c \times c$ 矩阵的行列式通常需要立方级的时间复杂度。如果在训练的每一步都评估此损失，计算成本将非常高昂。为了高效计算，权重矩阵 $W$ 使用其 LU 分解进行参数化：

$W = P L (U + \text{diag}(s))$

这里，$P$ 是固定的置换矩阵，$L$ 是主对角线全为 1 的下三角矩阵，$U$ 是主对角线全为 0 的上三角矩阵，$s$ 是一个向量 (vector)。因为三角矩阵的行列式是其对角线元素的乘积，所以 $W$ 的行列式对数可以简化为 $s$ 绝对值的对数之和：

$\log |\det(W)| = \sum_{i} \log |s_i|$

这显著降低了前向和反向传播 (backpropagation)过程中的计算开销。

现代耦合架构中的标准步骤是按顺序结合这三个组件。首先，ActNorm 对激活值进行归一化；接着，可逆 $1 \times 1$ 卷积混合通道；最后，仿射耦合层应用非线性变换。

> Glow 架构单个步骤中的操作序列。
