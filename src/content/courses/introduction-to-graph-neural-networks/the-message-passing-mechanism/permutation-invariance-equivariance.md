---
course: "introduction-to-graph-neural-networks"
chapter: "the-message-passing-mechanism"
lesson: "permutation-invariance-equivariance"
sourceId: 7623
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/permutation-invariance-equivariance"
title: "置换不变性与置换等变性"
description: "关于 GNN 为何必须对节点邻居的顺序保持不变性以及如何实现该属性的技术说明。"
order: 5
plots: []
sourceHash: "c4b3fe1ba575416c033c9725bf48543d36f5d71d5c8c263e3e67dafb5b3b8212"
sourceCorrections: []
---

图的一个显著特征是，与图像中的像素或句子中的单词不同，其节点没有天然的顺序。如果你询问一个节点的邻居，你得到的集合 `{u_1, u_2, u_3}` 与集合 `{u_3, u_1, u_2}` 是完全一样的。这种缺乏规范排序的特性给标准神经网络 (neural network)（如多层感知机 MLPs）带来了巨大的挑战，因为 MLPs 要求输入特征必须是固定且有序的向量 (vector)。如果你将邻居节点的特征输入到 MLP 中，输出会随着你选择的任意顺序而改变，从而导致结果不一致且毫无意义。

为了在图上有效运行，神经网络层必须遵循这一基本属性。这通过两个相关的原则来实现：置换不变性（Permutation Invariance）和置换等变性（Permutation Equivariance）。

### 聚合中的置换不变性

首先考虑消息传递公式中的 `AGGREGATE` 函数。它的任务是接收来自节点邻居的特征向量 (vector)集合 `{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \}`，并将它们合并为一个单一的消息向量 $\mathbf{m}_{\mathcal{N}(v)}$。

为了使网络无论邻居如何排序都能产生一致的输出，聚合函数必须具有**置换不变性**。如果一个函数 $f$ 的输出在输入顺序打乱时保持不变，那么该函数就是置换不变的。


$$
f(x_1, x_2, \dots, x_n) = f(x_{\pi(1)}, x_{\pi(2)}, \dots, x_{\pi(n)})
$$


这里，$\pi$ 是索引序列 $\{1, 2, \dots, n\}$ 的任意置换。

我们在前一节讨论的聚合函数，如 `sum`（求和）、`mean`（均值）和 `max`（最大值），都具备这种性质。例如，无论你从哪个邻居开始，对邻居特征向量求和的结果都是一样的。

- **求和 (Sum):** $\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}$
- **均值 (Mean):** $\frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}$
- **最大值 (Max):** $\max_{u \in \mathcal{N}(v)} (\mathbf{h}_u^{(l)})$（逐元素最大值）

通过在聚合中使用置换不变函数，我们确保从节点邻域生成的消息仅基于邻居的**集合**，而不是基于我们可能施加的任何人工排序。

### GNN 层中的置换等变性

虽然聚合步骤对单个节点的邻居顺序是不变的，但当考虑到图中的所有节点时，整个 GNN 层表现出一种略有不同的属性：**置换等变性**。

如果对输入的置换会导致输出发生同样的置换，那么该函数就是置换等变的。假设我们有一个函数 $F$，它接收一组节点特征 $\mathbf{H}$（每一行是一个节点特征向量 (vector)的矩阵）并输出一组新的特征 $\mathbf{H}'$。如果我们置换输入图中的节点（这对应于置换 $\mathbf{H}$ 的行以及邻接矩阵的行和列），输出特征 $\mathbf{H}'$ 将以完全相同的方式进行置换。

形式上，如果 $\mathbf{P}$ 是任何重新排列节点的置换矩阵，那么当函数 $F$ 满足以下条件时，它是等变的：


$$
F(\mathbf{P} \mathbf{A} \mathbf{P}^T, \mathbf{P} \mathbf{H}) = \mathbf{P} F(\mathbf{A}, \mathbf{H})
$$


这里，$\mathbf{A}$ 是邻接矩阵。操作 $\mathbf{P} \mathbf{A} \mathbf{P}^T$ 根据置换重新排列邻接矩阵，$\mathbf{P} \mathbf{H}$ 重新排列特征矩阵。该等式表明，先置换图的输入再应用 GNN 层，与先应用 GNN 层再置换其输出的效果是相同的。

消息传递机制自然地实现了这一点。每个节点的新表示 $\mathbf{h}_v^{(l+1)}$ 是根据其自身状态和来自其邻居的聚合消息计算得出的。由于这种计算是对每个节点并行执行的，如果我们重新索引图的节点，输出的嵌入 (embedding)向量也将以相同的方式重新索引。节点 `A` 的嵌入仍然是节点 `A` 的嵌入，即使我们现在将其称为节点 `B`。

> 图示说明不变性和等变性。对于不变性，输入到 `AGGREGATE` 函数的邻居特征的不同排列会产生相同的输出消息。对于等变性，在经过 `GNN 层` 之前置换图中所有节点的顺序，会导致输出节点表示集发生同样的置换。

### 为什么这些性质非常重要

置换不变性和等变性不仅仅是理论上的属性。它们是 GNN 能够学习依赖于图的基础结构（即拓扑结构）而非任意节点顺序的函数的原因。不具备置换等变性的模型可能会学到“列表中的第一个节点很重要”这种规律，而这种规律是毫无意义且无法泛化的。

通过遵循这些原则，GNN 可以：

1. **处理任意结构和大小的图：** 消息传递操作是针对每个节点局部定义的，使其独立于节点总数或它们的特定排列方式。
2. **学习可泛化的模式：** 模型基于邻域结构（例如，节点作为“中心”的角色或作为社区之间的“桥梁”）来学习特征，即使节点标签或存储顺序发生变化，这些特征也是一致的。

从本质上讲，这些属性确保了 GNN 学习的是图本身的信息，而不是我们在内存中记录图的方式。这对于构建强大且可靠的图结构数据模型至关重要。

## 参考资料

- [Neural Message Passing for Quantum Chemistry](http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf) — Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: Volume 70; Pages: 1263-1272; DOI: [10.5555/3305891.3305943](https://doi.org/10.5555/3305891.3305943)
  本文介绍了神经消息传递框架，形式化了GNN如何聚合信息，并隐式实现了置换不变性和等变性。
- [Geometric Deep Learning: Grids, Graphs, Groups, Geodesics, and Gauges](https://geometricdeeplearning.com/) — Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković (2021)
  Publisher: Cambridge University Press
  这本奠基性著作提供了一个统一的几何数据深度学习数学框架，严谨地阐述了对称性、不变性和等变性作为GNN的基本原则。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14; Pages: 1-159; DOI: [10.2200/S00996ED1V01Y202002AIM003](https://doi.org/10.2200/S00996ED1V01Y202002AIM003)
  这本教材全面介绍了图表示学习，其中详细解释了消息传递机制及其置换不变性和等变性。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  这篇开创性论文介绍了图卷积网络（GCNs），这是一种广泛采用的GNN模型，它体现了消息传递范式及其固有的置换不变性和等变性。
