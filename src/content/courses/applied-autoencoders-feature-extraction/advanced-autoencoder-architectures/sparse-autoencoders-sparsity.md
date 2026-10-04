---
course: "applied-autoencoders-feature-extraction"
chapter: "advanced-autoencoder-architectures"
lesson: "sparse-autoencoders-sparsity"
sourceId: 6464
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-4-advanced-autoencoder-architectures/sparse-autoencoders-sparsity"
title: "稀疏自编码器：实现表示的稀疏化"
description: "了解稀疏自编码器以及它们如何促使稀疏潜在表示以实现更好的特征解耦。"
order: 1
plots: []
sourceHash: "c7ee37c9c0bf9e9a9587f4e2e7a2897ec5fe12b43e643098ef48fd7e34c828c8"
sourceCorrections: []
---

虽然基础自编码器在学习压缩表示方面表现出色，但它们的隐藏层可能无法总是捕获高度专业化或解耦的特征。尤其是在*过完备*自编码器中，其隐藏层神经元数量多于输入层，网络存在只学习一个恒等函数、将输入复制到输出而没有找出有意义的底层数据结构的风险。稀疏自编码器通过对隐藏层的激活施加“稀疏性约束”来解决此问题。

### 自编码器中的稀疏性是什么？

在自编码器的情形下，稀疏性指的是对于任何给定的输入样本，只有一小部分隐藏层神经元是“活跃的”，这意味着它们的输出值明显非零。大多数隐藏神经元保持不活跃或接近零。

可以把隐藏层想象成一个专家小组。在一个稀疏系统中，当接收到特定信息（一个输入样本）时，只有少数与该信息专业高度相关的专家会“发言”（变得活跃）。其他专家保持沉默。这与密集表示形成对比，在密集表示中，许多神经元可能会响应任何给定输入。

> 隐藏层激活模式。左图：密集模式，其中许多神经元处于活跃状态。右图：稀疏模式，由稀疏自编码器促成，其中对于给定输入，只有少数神经元高度活跃（红色表示活跃，灰色表示不活跃）。

### 为何寻求稀疏性？其益处

强制实行稀疏性对特征学习有多种益处：

1. **学习专门化特征：** 当一次只能有少数神经元活跃时，每个神经元都会被促使专门识别输入数据中更具体的模式或属性。这能产生更细致且可解释的特征。
2. **处理过完备表示：** 如果隐藏层大于输入层（一个过完备自编码器），稀疏性会阻止自编码器仅仅学习恒等函数。即使有许多可用神经元，该约束也会迫使网络找出一种表示，其中对于任何输入只使用一小部分，从而有效地创建一个基于激活而非仅层大小的信息瓶颈。
3. **解耦特征的可能性：** 稀疏性可以促使模型学习更独立或“解耦”的特征，其中不同的隐藏单元对应数据中不同的底层变化因素。
4. **泛化能力的提升：** 通过对每个输入聚焦于一小组高度相关的特征，稀疏自编码器有时能更好地泛化到新的、未见过的数据，因为它们可能较不容易记忆训练集中的噪声或不相关细节。

### 实现稀疏性的方法

稀疏性通常通过在自编码器的主要损失函数 (loss function)中添加一个*稀疏性惩罚项*来实现。这样，总损失函数就成为重构误差（例如，均方误差）与该稀疏性惩罚的组合：

$L_{total} = L_{reconstruction} + \lambda \cdot P_{sparsity}$

这里，$\lambda$ (lambda) 是一个超参数 (parameter) (hyperparameter)，用于控制稀疏性惩罚相对于重构损失的权重 (weight)或重要性。定义 $P_{sparsity}$ 的两种常见方法是L1正则化 (regularization)和 Kullback-Leibler (KL) 散度。

#### 1. 激活上的L1正则化

该方法添加一个与隐藏层激活绝对值之和成比例的惩罚。如果 $h_j$ 是给定输入下隐藏层中第 $j$ 个神经元的激活，那么L1稀疏性惩罚为：


$$
P_{L1} = \sum_{j} |h_j|
$$


L1惩罚促使许多激活 $h_j$ 变为恰好为零或非常接近零。这类似于线性模型中对权重进行L1正则化如何促进稀疏权重向量 (vector)。通过惩罚较大的激活，网络学习到对于任何特定输入只使用少数具有强激活的隐藏单元。

#### 2. KL散度惩罚

一种更具统计学依据的实现稀疏性的方法是使用Kullback-Leibler (KL) 散度。该方法旨在使每个隐藏神经元在一批训练数据上的*平均激活*接近一个小的目标值，通常表示为 $\rho$ (rho)。例如，我们可以设置 $\rho = 0.05$，这意味着我们希望每个隐藏神经元平均只在5%的训练样本上活跃。

令 $\hat{\rho}_j$ 为隐藏神经元 $j$ 在一批 $m$ 个训练样本上计算的实际平均激活：


$$
\hat{\rho}_j = \frac{1}{m} \sum_{k=1}^{m} h_j(x^{(k)})
$$


其中 $h_j(x^{(k)})$ 是神经元 $j$ 对第 $k$ 个训练样本 $x^{(k)}$ 的激活。

单个神经元 $j$ 的期望平均激活 $\rho$ 与观测平均激活 $\hat{\rho}_j$ 之间的KL散度（假设激活值在0到1之间，例如经过sigmoid函数后）由下式给出：


$$
\text{KL}(\rho || \hat{\rho}_j) = \rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}
$$


该项衡量了两种分布（期望均值为 $\rho$ 的伯努利分布和观测均值为 $\hat{\rho}_j$ 的分布）之间的“距离”或散度。总稀疏性惩罚是所有 $S_h$ 个隐藏神经元的这些KL散度项的总和，再乘以一个加权因子 $\beta$：


$$
P_{KL} = \beta \sum_{j=1}^{S_h} \text{KL}(\rho || \hat{\rho}_j)
$$


通过将此惩罚项作为总损失的一部分进行最小化，自编码器会调整其权重，使得每个隐藏神经元 $j$ 的平均激活 $\hat{\rho}_j$ 更接近目标稀疏性参数 $\rho$。

### 将稀疏表示用作特征

一旦训练好稀疏自编码器，其隐藏层对给定输入的激活就会提供该输入的稀疏表示。这种稀疏编码，即隐藏单元激活向量 (vector)，可以被提取并用作下游机器学习 (machine learning)任务（例如分类或聚类）的特征。这些特征通常更为专门化，并且由于其集中性，可以提升性能。

总而言之，稀疏自编码器通过明确惩罚隐藏层中的密集激活，提供了一种引导自编码器学习更具体且可能解耦特征的方法。这使得它们成为一种有价值的工具，特别是在处理过完备架构或旨在获得高度选择性特征检测器时。L1和KL散度之间的选择，以及调整它们各自的超参数 (parameter) (hyperparameter)，可以灵活地控制所需的稀疏性特点。

## 参考资料

- [Sparse Autoencoders](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) — Andrew Ng and the UFLDL team (2013)
  Publisher: Stanford University
  涵盖稀疏自编码器的核心原理，包括诱导稀疏性的KL散度惩罚的解释。这是一个主要的在线教程。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7 (Regularization for Deep Learning), Chapter 14 (Autoencoders)
  一本权威教材，涵盖自编码器、L1等正则化技术以及深度学习理论，为稀疏性提供背景。
- [Emergence of simple-cell receptive field properties by learning a sparse code for natural images](https://www.nature.com/articles/381607a0) — Bruno A. Olshausen and David J. Field (1996)
  Journal: Nature; Volume: 381; Pages: 607-609; DOI: [10.1038/381607a0](https://doi.org/10.1038/381607a0)
  一篇开创性论文，展示了稀疏编码如何导致专业化、生物学上合理的特征的出现，阐释了稀疏表示的优点。
