---
course: "deep-learning-regularization-optimization"
chapter: "dropout-regularization"
lesson: "dropout-training-mechanism"
sourceId: 4927
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/dropout-training-mechanism"
title: "Dropout 在训练时的工作原理"
description: "详细说明在训练阶段应用 Dropout 的过程。"
order: 2
plots: []
sourceHash: "86daa21254a6a48f808e9c01fb5c0fcc27736a0e5087b2b4b49aaa7c9f75152b"
sourceCorrections: []
---

Dropout 在训练时将随机性直接引入网络结构。该方法与修改损失函数 (loss function)的正则化 (regularization)技术（例如权重 (weight)正则化）形成对比。Dropout 在每次前向传播时修改网络本身。

主要思路直接明了：对于每个训练样本（或小批量），以一定的概率 $p$ 从网络中临时移除单元（及其连接）。每个单元都有独立的 $p$ 概率被舍弃，这意味着其输出在那个特定的前向和反向传播 (backpropagation)中被设为零。

### Dropout 掩码

设想你的网络中有一个包含 $N$ 个单元的层。在训练迭代的前向传播中，我们生成一个随机二元掩码，称之为 $m$，其大小与该层的输出相同。此掩码中的每个元素 $m_i$ 都来自一个伯努利分布：以 $1-p$ 的概率（“保留概率”）为 1，以 $p$ 的概率（“舍弃概率”）为 0。


$$
m_i \sim \text{Bernoulli}(1-p)
$$


如果单元对应的掩码值 $m_i$ 为 1，则该单元正常运行。如果 $m_i$ 为 0，则该单元的输出在此特定的前向传播中被强制设为零。然后，此掩码按元素方式应用于层的激活输出，例如 $a$。结果 $a_{dropout}$ 随后传递到下一层。


$$
a_{dropout} = a \odot m \quad (\text{其中 } \odot \text{ 表示元素乘法})
$$


重要的一点是，每次向网络呈现训练样本或小批量时，都会生成一个*不同*的掩码 $m$。网络这种持续的“稀疏化”有助于避免单元变得过度专业化或过度依赖特定其他单元的存在。

> 一个包含四个单元的层在两个不同训练步骤中经历 Dropout。带虚线轮廓的灰色单元在该特定步骤中被随机设为零输出（已舍弃）。请注意，被舍弃的单元集合在不同步骤之间会发生变化。

### 对训练的影响

此过程具有显著作用：

1. **强制分布式表示：** 由于任何单元都可能随机消失，网络不能过度依赖任何一个单元。它会学习到一些冗余的特征，这些特征分布在多个单元中。如果一个路径被“舍弃”，其他路径可能有机会进行弥补。
2. **集成效应（近似）：** 用 Dropout 训练网络可以被视为隐式训练大量共享权重 (weight)的“稀疏化”网络。每个训练步骤都使用一个不同的稀疏化网络。虽然这不是真正的集成平均，但这个过程提供了相似的正则化 (regularization)益处。
3. **减少协同适应：** 神经元对其他单个神经元的特定噪声激活变得不那么敏感，从而减少可能仅适合训练数据的复杂协同适应。

### 反向传播 (backpropagation)的考虑

在反向传播过程中，梯度只通过*未*被舍弃的单元（即 $m_i = 1$ 的单元）进行计算和传播。输出被设为零的单元不参与该特定训练步骤的梯度计算，实际上也被临时从梯度计算路径中移除。

训练中这种持续的单元洗牌和禁用是 Dropout 帮助防止过拟合 (overfitting)的主要机制。然而，这也意味着我们需要一种一致的方法在推断或测试时使用*完整*的网络，这一点我们将在下一节关于激活缩放中进行讨论。

## 参考资料

- [Dropout: A simple way to prevent neural networks from overfitting](https://api.semanticscholar.org/CorpusID:6844431) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  这篇基础性论文介绍了Dropout技术，详细阐述了其机制并展示了其在缓解神经网络过拟合方面的有效性。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7, Section 7.12
  一本权威教材，其中有专门的Dropout章节，严格解释了其理论基础和正则化实践。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Regularization](https://cs231n.github.io/neural-networks-2/#reg) — Stanford University (2017)
  知名计算机视觉课程的讲义，提供了对Dropout作为神经网络正则化策略的清晰直观解释。
