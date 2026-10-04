---
course: "meta-learning-foundation-models"
chapter: "advanced-metric-based-meta-learning"
lesson: "prototypical-networks-revisited"
sourceId: 4324
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-3-advanced-metric-based-meta-learning/prototypical-networks-revisited"
title: "原型网络再探讨"
description: "嵌入函数、距离度量和归纳偏置的更详细讨论。"
order: 1
plots: []
sourceHash: "11c75c961d5d0ab2546ffe3144c140244f1fa029160bcb8cdffc4618bb3738b4"
sourceCorrections: []
---

原型网络在基于度量的元学习中提供了一种巧妙的方法，其核心思想是为任务支持集中的每个类别创建有代表性的*原型*。查询点的分类取决于在学习到的嵌入 (embedding)空间$f_\phi(\cdot)$中找到最近的类别原型。尽管基本原理直接明了，但其有效应用，尤其是在使用基础模型的深维嵌入时，需要更细致地查看其核心组成部分。

### 嵌入 (embedding)函数 $f_\phi(\cdot)$

原型网络的成功极其依赖于嵌入函数$f_\phi$的质量。此函数将输入数据点$x$映射到嵌入空间$\mathbb{R}^D$中的向量 (vector)$z = f_\phi(x)$。目标是使这个空间呈现一种结构，即同一类别的点聚集在一起，而不同类别的点则良好分离。

在基础模型（如大型视觉Transformer或语言模型）的背景下，$f_\phi$通常直接来源于预训练 (pre-training)模型的中间或最终层表示。这借助了在大规模预训练中学习到的丰富通用特征。重要考量包括：

1. **层选择：** 来自不同层的嵌入捕获不同层次的抽象信息。较早的层可能保留更多细粒度细节，而较深的层捕获更多语义信息。最佳选择通常依赖于任务并需要经验验证。
2. **固定与微调 (fine-tuning)嵌入器：**
   - **固定：** 纯粹将基础模型作为固定的特征提取器（$\phi$在元训练期间不更新）是计算高效的。其成功依赖于预训练空间已适合通过原型进行少样本分类的假设。
   - **微调：** 在元训练期间更新$\phi$可以使嵌入空间专门适应各种任务的原型分类。这能带来更好的性能，但会显著增加计算成本，并需要仔细优化，尤其考虑到基础模型的规模。第5章讨论的参数 (parameter)高效微调（PEFT）等技术可能在元训练期间应用于$\phi$。

### 原型计算

对于特定任务$\mathcal{T}$和类别$k$，原型$c_k$通常计算为属于该类别的嵌入 (embedding)支持样本$\{x_i, y_i\}_{i=1}^{N_S}$的均值向量 (vector)：


$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)
$$


其中 $S_k = \{(x_i, y_i) \in S \mid y_i = k\}$ 是支持集$S$中属于类别$k$的子集。

> 原型网络的流程：支持集样本被嵌入，然后取均值形成类别原型。嵌入的查询点根据其与这些原型的距离进行分类。

这种均值聚合作为一种简单的降噪和概括形式。然而，在极低样本场景（例如，单样本学习）中，原型就是单个支持样本的嵌入，这使其对异常值或非典型样本敏感。尽管存在其他聚合方法（例如，加权均值、强健估计器），但标准均值因其简单性以及在嵌入空间结构良好时的有效性而仍然普遍。

### 距离度量

用于比较嵌入 (embedding)的查询点$z_q = f_\phi(x_q)$与类别原型$c_k$的距离函数$d(\cdot, \cdot)$的选择是另一个重要的设计决策。

1. **欧氏距离平方 ($L_2^2$)：** 这是最常见的选择，常用于原始公式中。查询点$x_q$属于类别$k$的概率通过对负距离进行softmax来建模：
   
   $$
   
   $$
   

p(y=k | x\_q) = \frac{\exp(-d(z\_q, c\_k))}{\sum\_{k'} \exp(-d(z\_q, c\_{k'}))} = \frac{\exp(-|f\_\phi(x\_q) - c\_k|^2\_2)}{\sum\_{k'} \exp(-|f\_\phi(x\_q) - c\_{k'}|^2\_2)}

2. \*\*余弦距离：\*\* 在高维空间中，这对于基础模型嵌入 ($D \gg 1000$) 来说很常见，欧氏距离可能存在问题，因为存在度量集中现象，即点之间的距离倾向于变得区分度较低。余弦相似度，或其对应的距离（$d_{cos}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v|}$），侧重于向量之间的角度而非其大小。这在高维环境中可以更强健，并且在处理向量大小可能不包含重要类别特定信息或变化很大的嵌入时，通常更受欢迎。使用余弦距离隐式地涉及对嵌入和原型的L2范数归一化。
欧氏距离和余弦距离的选择通常取决于$f_\phi$生成的嵌入空间的特性，以及向量大小对于区分是否具有意义。
### 归纳偏置
原型网络具有很强的归纳偏置：它们假设每个类别可以由嵌入空间中的单个点（原型）充分表示，并且当使用欧氏距离时，类别在这些原型周围大致形成超球形簇。如果嵌入函数$f_\phi$被训练（或者在预训练模型的情况下，固有地具有这种结构），将类别映射到此类良好分离、紧凑的区域，则效果良好。
当使用基础模型的嵌入时，这种偏置可能并非总是与特定下游少样本任务的数据结构完全符合，尤其是在任务涉及具有显著类内方差或多模态分布的类别时。其有效性取决于基础模型学习到的通用表示是否恰好能很好地对类别进行聚类，或者元训练$f_\phi$是否成功施加了这种结构。
### 与深度度量学习的关联
训练嵌入函数$f_\phi$实际上是一个深度度量学习问题。目标是学习一种转换，使同一类别的样本更接近，并使不同类别的样本更远离，具体而言，以支持基于原型的分类的方式。尽管原型网络定义了分类机制（均值原型 + 距离），但元学习期间$f_\phi$的训练通常隐式或显式地优化度量学习目标。例如，从对原型距离进行softmax导出的损失函数鼓励必要的聚类结构。更显式的度量学习损失，例如三元组损失（在3.4节中讨论），也可以被纳入元训练过程，以更直接地塑造嵌入空间。
重新探讨原型网络显示，其简单性掩盖了关于嵌入函数、原型计算和距离度量的重要潜在选择。这些选择在扩展到基础模型常见的深维表示时变得尤为重要，需要仔细考量计算限制、归纳偏置以及嵌入空间的几何属性。

## 参考资料

- [Prototypical Networks for Few-Shot Learning](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6766d11e7287e07bee655be2772-Abstract.html) — Jake Snell, Kevin Swersky, Richard Zemel (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; DOI: [10.55982/annals-2017-384](https://doi.org/10.55982/annals-2017-384)
  介绍原型网络的原始论文，定义了其核心算法及其在少样本学习中的应用。
- [A Survey of Meta-Learning in Few-Shot Learning](https://ieeexplore.ieee.org/document/9132104) — Yaqing Wang, Quanming Yao, James T. Kwok, Lionel M. Ni (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 3211-3235; DOI: [10.1109/TPAMI.2020.3006434](https://doi.org/10.1109/TPAMI.2020.3006434)
  全面概述了少样本学习中的元学习方法，包括像原型网络这样的基于度量的方法。
