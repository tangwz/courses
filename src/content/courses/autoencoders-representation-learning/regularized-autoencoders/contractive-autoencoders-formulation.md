---
course: "autoencoders-representation-learning"
chapter: "regularized-autoencoders"
lesson: "contractive-autoencoders-formulation"
sourceId: 2833
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders/contractive-autoencoders-formulation"
title: "收缩自编码器公式"
description: "了解收缩自编码器如何惩罚所学特征对输入变化的敏感性。"
order: 4
plots: []
sourceHash: "1b9861e0b01c2c63fd4ae0840927209b81261baec3ad015b781c6a08b1bd86ca"
sourceCorrections: []
---

收缩自编码器（CAEs）开发对微小输入变化具有鲁棒性的表示。这些模型明确旨在使所学特征映射（由编码器函数 $f(x)$ 表示）对输入数据 $x$ 的微小变化不敏感。这与其他自编码器类型不同，例如稀疏自编码器侧重于通过将不重要的潜在单元置零来寻找简洁表示，而去噪自编码器则通过从损坏版本重建数据来学习。

核心思想是*收缩*。我们希望自编码器学习一种映射，使得输入空间中的微小变化对应于潜在特征空间中更小的变化。这促使模型捕捉代表数据中重要变化因素的特征，同时有效地忽略微小、不相关的波动或噪声。设想编码器将高维输入空间中的点映射到低维潜在空间；收缩映射确保输入空间中的邻域在映射到潜在空间时会收缩。

### 数学表述

为实现这种不敏感性，CAEs 在标准重建损失中添加一个惩罚项。此惩罚衡量编码器输出 $h = f(x)$ 响应输入 $x$ 的无穷小变化而变化的程度。这种敏感性由编码器函数 $f$ 关于 $x$ 的雅可比矩阵捕获。

雅可比矩阵 $J(x)$ 是一个矩阵，包含编码器输出向量 (vector) $h$ 相对于输入向量 $x$ 的所有一阶偏导数：


$$
J_{ij}(x) = \frac{\partial h_i}{\partial x_j}
$$


其中 $h_i$ 是潜在表示的第 $i$ 个元素，$x_j$ 是输入的第 $j$ 个元素。

收缩自编码器添加的惩罚与该雅可比矩阵的平方弗罗贝尼乌斯范数成比例。矩阵 $A$ 的弗罗贝尼乌斯范数 $||A||_F$ 定义为 $||A||_F = \sqrt{\sum_{i,j} A_{ij}^2}$。因此，惩罚项为：


$$
\Omega_{CAE}(x) = ||J(x)||_F^2 = \sum_{i,j} \left( \frac{\partial h_i(x)}{\partial x_j} \right)^2
$$


该项对潜在特征相对于输入的所有偏导数的平方求和。最小化此项促使这些导数变小，使映射 $f(x)$ 具有收缩性。

收缩自编码器的完整损失函数 (loss function)是重建损失（例如，均方误差或二元交叉熵）与此收缩惩罚之和，由超参数 (parameter) (hyperparameter) $\lambda$ 加权：


$$
L_{CAE}(x, \hat{x}) = L_{rec}(x, \hat{x}) + \lambda \Omega_{CAE}(x)
$$


$$
L_{CAE}(x, \hat{x}) = L_{rec}(x, \hat{x}) + \lambda \sum_{i,j} \left( \frac{\partial h_i(x)}{\partial x_j} \right)^2
$$


其中，$\hat{x} = g(f(x))$ 是重建输出，$L_{rec}$ 是重建损失，$\lambda > 0$ 控制收缩惩罚的强度。较大的 $\lambda$ 会强制更强的收缩，但如果设置过高，可能会影响重建质量。调整 $\lambda$ 有助于平衡学习不敏感表示与准确重建输入数据之间的权衡。

### 解释与流形学习的联系

最小化雅可比范数促使编码器 $f(x)$ 主要在训练数据点附近具有收缩性。这与流形学习存在有趣的联系。如果数据位于高维输入空间中的低维流形上或附近，CAE 倾向于学习捕获沿流形变化的特征，同时收缩（对）与流形*正交*的方向（不敏感）。本质上，该惩罚促使编码器局部学习数据流形的切空间。

> 收缩映射的示意图。输入空间中微小的扰动 $\delta x$ 应该导致潜在空间中更小的扰动 $\delta h$，这是通过最小化编码器 $f$ 的雅可比范数实现的。

### 实现考量

计算完整的雅可比矩阵及其弗罗贝尼乌斯范数会给训练过程带来显著的计算开销，特别是对于高维输入和潜在空间。幸运的是，现代深度学习 (deep learning)框架凭借自动微分能力可以高效地计算这种惩罚。虽然精确计算可能仍比简单的重建损失慢，但对于典型网络规模而言通常是可行的。框架通常提供优化的方法来计算平方梯度和或使用近似方法，避免显式构建完整的雅可比矩阵。

### 与去噪自编码器的比较

收缩自编码器和去噪自编码器都旨在学习对输入变化不太敏感的表示。然而，它们实现此目标的方式不同：

- **DAEs:** *隐式地*学习鲁棒性，通过训练模型从随机损坏的版本重建原始输入。模型必须学习足够稳定的特征，以忽略训练期间引入的特定类型噪声。
- **CAEs:** *显式地*学习鲁棒性，通过添加惩罚项直接最小化所学特征相对于输入扰动*所有*方向的敏感性。

DAE 和 CAE 之间的选择可能取决于具体任务和数据中预期变化的性质。CAE 提供了一种更直接的分析方式来强制不敏感性，而 DAE 可能更直观，有时根据所选噪声模型更易于实现。

总之，收缩自编码器提供了一种数学上有依据的正则化 (regularization)方法，通过惩罚编码器映射的敏感性。这促使模型学习捕捉数据核心结构，同时对微小的局部扰动保持不变性的表示，有助于更好的特征学习。

## 参考资料

- [Contractive Auto-Encoders: Explicitly Promoting Small Derivatives of the Code Function](https://www.jmlr.org/papers/v15/rifai11a/rifai11a.pdf) — Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio (2011)
  Journal: JMLR Workshop and Conference Proceedings; Publisher: JMLR; Volume: 15; Pages: 1089-1096
  介绍收缩自编码器及其数学公式（包括雅可比惩罚）的基础论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教科书，在深度学习的广泛背景下涵盖了自编码器，包括收缩自编码器及其理论基础。
- [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538) — Yoshua Bengio, Aaron Courville, and Pascal Vincent (2013)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Volume: 35; Pages: 1798-1828; DOI: [10.48550/arXiv.1206.5538](https://doi.org/10.48550/arXiv.1206.5538)
  一篇综述文章，讨论了各种表示学习技术，包括收缩自编码器，以及它们与流形学习等概念的关系。
