---
course: "introduction-to-graph-neural-networks"
chapter: "foundational-gnn-architectures"
lesson: "spatial-interpretation-graph-convolutions"
sourceId: 7627
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/spatial-interpretation-graph-convolutions"
title: "图卷积的空间物理解读"
description: "从空间视角观察图卷积，此时操作是直接在图结构上进行的。"
order: 2
plots: []
sourceHash: "ae390de7285df2d643b9059da8d5be5c55ea0dce8602e13303ea0abb79e656c5"
sourceCorrections: []
---

虽然图卷积网络（GCN）的数学起源可以追溯到谱图理论（涉及图傅里叶变换等复杂运算），但更直观、更容易理解的方法是从空间视角来看待它。这种观点将图卷积视为直接在图结构上进行的消传递操作，这与邻域聚合框架完全一致。

### 从图像卷积到图卷积

为了建立直观认知，可以考虑卷积神经网络 (neural network)（CNN）在图像上的操作方式。CNN 使用一个小核或过滤器在像素网格上滑动。在每个位置，卷积核计算其结构化邻域内像素值的加权平均。这个过程有效地聚合了局部信息，为每个像素生成新的特征表示。

GCN 执行类似的功能，但在不规则、非结构化的图上运行。与图像中固定大小的邻域网格不同，GCN 中每个节点的邻域由其相连的边定义。针对某个节点的“卷积”操作包括聚合来自其邻居的特征向量 (vector)。

在最简单的形式中，GCN 层通过取邻居特征向量的平均值来更新节点的特征向量。这种操作直接反映了 CNN 卷积核的目标：根据局部环境的特征为某个点创建新的表示。

### GCN 公式以节点为中心的视角

我们可以将 GCN 层的操作看作是针对每个节点进行的处理，而不仅仅是单一的矩阵乘法。单个节点 $i$ 的更新可以理解为一个“聚合与更新”的过程：

1. **聚合邻居信息：** 模型收集来自所有相邻节点 $j$（包括通过自环连接的节点 $i$ 本身）的特征向量 (vector) $h_j$。
2. **归一化 (normalization)聚合结果：** 聚合后的总和随后被归一化。这是非常的一步。如果没有这一步，拥有许多邻居的节点（高度数节点）在聚合后特征向量的量级会大得多，从而导致训练不稳定。GCN 特有的归一化方案同时利用了中心节点 $i$ 和邻居节点 $j$ 的度数。

第 $l+1$ 层中节点 $i$ 的隐层表示 $h_i^{(l+1)}$ 的更新规则如下：


$$
h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\text{deg}(i)\text{deg}(j)}} h_j^{(l)} W^{(l)} \right)
$$


其中：

- $\mathcal{N}(i)$ 是节点 $i$ 的邻居集合。
- $\text{deg}(i)$ 是节点 $i$ 的度数。
- $W^{(l)}$ 是第 $l$ 层的可训练权重 (weight)矩阵，由所有节点共享。
- $\sigma$ 是非线性激活函数 (activation function)，例如 ReLU。

这个公式直接将空间想法转化为数学。对于每个节点 $i$，我们遍历其邻居 $j$，获取它们当前的特征向量 $h_j^{(l)}$，通过权重矩阵 $W^{(l)}$ 进行变换，然后根据节点度数确定的权重进行加权求和。

> 以节点为中心的图卷积视图。节点 A 通过聚合来自其邻居（B、C、D）和自身的变换后的特征向量 (h) 来更新其表示。

### 归一化 (normalization)的意义

归一化项 $\frac{1}{\sqrt{\text{deg}(i)\text{deg}(j)}}$ 是 GCN 模型的显著特征。虽然简单地按照中心节点的度数取平均值（$\frac{1}{\text{deg}(i)}$）似乎也行，但这种对称归一化在实际应用中表现更好。它在消息传递步骤中兼顾了源节点和目标节点的度数，防止特征向量 (vector)的尺度受到高度数节点过度影响。

通过从空间角度观察图卷积，我们可以看到 GCN 层是消息传递方案的一种高效且专门的实现。它为 `AGGREGATE` 函数（度数归一化求和）和 `UPDATE` 函数（应用非线性变换）定义了特定的选择。这种视角使得 GCN 更容易与 GraphSAGE 和 GAT 等其他架构进行比较，因为那些架构只是为这些函数提供了不同的选择。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  引入图卷积网络(GCN)及其带对称归一化特定更新规则的基础论文。
- [Neural Message Passing for Quantum Chemistry](http://proceedings.mlr.press/v70/gilmer17a.html) — Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1263-1272; DOI: [10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)
  形式化了消息传递神经网络（MPNN）框架，将GCNs作为消息传递的特定实例进行说明。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, and Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE框架，该框架扩展了GCNs之外的邻域聚合，并支持归纳学习。
