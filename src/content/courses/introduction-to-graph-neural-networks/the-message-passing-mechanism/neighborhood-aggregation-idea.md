---
course: "introduction-to-graph-neural-networks"
chapter: "the-message-passing-mechanism"
lesson: "neighborhood-aggregation-idea"
sourceId: 7619
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/neighborhood-aggregation-idea"
title: "邻域聚合思想"
description: "了解 GNN 的核心原理：节点如何聚合来自其局部邻居的信息以更新自身的表示。"
order: 1
plots: []
sourceHash: "2981ac50dd245ae27259c9064a9d516e7c3249adce25d2276ebe45966e576e62"
sourceCorrections: []
---

任何神经网络 (neural network)的核心都围绕着一个简单的问题：我们如何为给定的数据点计算出有用的特征表示？对于图像，卷积神经网络 (CNN) 在固定的像素网格上滑动卷积核，聚合局部信息以构建边缘、纹理和形状等特征。对于文本，循环神经网络 (RNN) 处理单词序列，并在每一步更新其状态。

但图数据呢？图中的节点不像像素那样具有固定的邻居网格，也不像句子那样具有有序的序列。一个节点可能有一个邻居，也可能有一千个，并且它们之间没有固定的排序。这就是图神经网络旨在解决的核心挑战。

GNN 的基本思想是：通过整合来自局部邻域的信息，可以丰富节点的特征。以社交网络为例，你个人的兴趣和行为往往反映了你朋友的兴趣。如果你想预测自己是否会喜欢一部新电影，了解你亲密的朋友是否喜欢它可能是一个非常强烈的信号。同样，在引用网络中，一篇研究论文的主题与其引用的论文主题密切相关。

GNN 通过一种称为**邻域聚合**或**消息传递**的过程将这种直觉转化为实际操作。我们不再孤立地观察一个节点，而是通过汇总其邻居的特征来为该节点创建一个新的、更强大的表示。

### 聚合过程

这个过程非常直接：对于目标节点，我们收集其所有直接邻居的特征向量 (vector)，并将它们合并为一个向量。这个合并后的向量充当了汇总整个邻域信息的“消息”。

例如，假设目标节点 *A* 连接到节点 *B*、*C* 和 *D*。这些节点中的每一个都有一个关联的特征向量，我们分别称之为 $\mathbf{h}_A$、$\mathbf{h}_B$、$\mathbf{h}_C$ 和 $\mathbf{h}_D$。GNN 层的第一步就是聚合来自 *A* 的邻居的信息。

> 收集邻居节点 B、C 和 D 的特征向量，并将其传递给聚合函数。该函数产生一个单一的汇总向量，即“消息”，然后发送到目标节点 A。

这种聚合必须能够处理节点邻域无序且大小可变的特性。我们不能简单地将特征向量拼接起来，因为那将依赖于邻居的任意排列顺序。相反，我们需要一种具有**置换不变性**的操作，这意味着无论输入的顺序如何，它都会产生相同的输出。

这种聚合函数的常见选择包括：

- **求和 (Sum)：** 简单地将所有邻居向量相加。
- **均值 (Mean)：** 计算所有邻居向量的逐元素平均值。
- **最大值 (Max)：** 取所有邻居向量的逐元素最大值。

通过应用像 `mean` 这样的操作，我们计算出的邻域汇总信息与邻居的数量或处理顺序无关。这个聚合向量代表了节点局部环境的集体信息。这是我们之前看到的通用 GNN 层公式的前半部分：


$$
\mathbf{m}_{\mathcal{N}(v)}^{(l)} = \text{聚合}^{(l)} \left( \{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \} \right)
$$


这里，$\mathbf{m}_{\mathcal{N}(v)}^{(l)}$ 是节点 $v$ 在第 $l$ 层的聚合消息，通过对来自所有邻居节点 $u$ 的特征向量集 ($\mathbf{h}_u^{(l)}$) 应用 `聚合` 函数计算得出。

现在我们有了一个简洁的邻域汇总信息，下一步就是使用这条消息来更新目标节点自身的表示。这是 GNN 层的第二个部分，我们将在下一节中详细查看。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  提出了一种广泛采用的图神经网络模型，通过使用局部谱滤波器近似来聚合邻居特征以学习节点表示。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE框架，该框架学习一个函数来聚合节点局部邻域的信息，从而实现对未见节点的归纳学习。
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/) — Jure Leskovec (2023)
  Publisher: Stanford University
  一门广受好评的大学课程，涵盖图神经网络的基础知识，包括邻域聚合和消息传递机制的详细解释。
