---
course: "graph-neural-networks-gnns"
chapter: "gnn-training-complexities"
lesson: "gnn-scalability-challenges"
sourceId: 2702
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/gnn-scalability-challenges"
title: "处理大型图：可扩展性挑战"
description: "讨论在大型图数据集上训练GNN时的内存和计算瓶颈。"
order: 4
plots: []
sourceHash: "7979a7974977128753893f4bfe2786f994a8fd95a2cb070d41fab084f7caf913"
sourceCorrections: []
---

训练图神经网络 (neural network)涉及如过平滑和过压缩等独特的难题。此外，在处理实际使用中常见的图时，会出现另一个重要的实际困难：它们庞大的规模。社交网络、网页图、引用网络、生物相互作用网络和知识图谱可以轻易包含数百万甚至数十亿的节点和边。将通常假设整个图及其特征可以同时处理的标准GNN训练方法应用于如此大规模的图上，是不可行的。

这种不可行性源于两个主要瓶颈：计算开销和内存需求。

### 计算瓶颈

一个标准GNN层在邻域间执行消息传递和聚合。设有一个$L$层GNN。为了计算所有节点的最终嵌入 (embedding)，信息理论上会传播至多$L$跳远。在一个典型的全图训练设置中，每个时期都需要处理图中的每个节点和边。

单个GNN层的计算复杂度通常涉及稀疏矩阵乘法等操作。例如，一个基本的图卷积网络（GCN）层使用如下公式，从前一层$H^{(l)}$更新节点表示$H^{(l+1)}$：


$$
H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})
$$


这里，$\hat{A}$是带自环的邻接矩阵，$\hat{D}$是对应的度矩阵，$H^{(l)}$是层$l$的$|V| \times d^{(l)}$特征矩阵（其中$|V|$是节点数，$d^{(l)}$是特征维度），而$W^{(l)}$是$d^{(l)} \times d^{(l+1)}$的权重 (weight)矩阵。核心计算涉及将（归一化 (normalization)）邻接矩阵与特征矩阵相乘。即使使用稀疏矩阵表示，此操作的开销也大致与边数$|E|$和特征维度成比例。一个$L$层GNN的每个时期的开销大约是$O(L \times |E| \times d)$，其中$d$代表典型的特征维度。

当$|E|$达到数十亿或数万亿时，这种计算变得极其耗时，使得训练即使在强大的硬件加速器上也慢到无法接受。每次梯度更新都需要对这个庞大结构进行一次完整遍历。

### 内存瓶颈

内存限制，特别是GPU内存（VRAM），可能比计算时间更具限制性。以全图方式训练GNN需要存储几个大型数据结构：

1. **图结构：** 图的邻接信息（例如，以坐标格式（COO）或压缩稀疏行格式（CSR））。尽管稀疏格式远优于密集表示（$O(|V|^2)$），但存储数十亿条边的连接性仍然需要大量内存（$O(|E|)$或$O(|V| + |E|)$）。
2. **节点特征：** 所有节点的输入特征（$|V| \times d_{in}$）。对于大的$|V|$，仅此矩阵就可能超出可用的RAM或VRAM。
3. **中间嵌入 (embedding)（激活）：** 这通常是最主要的内存限制。在前向传播期间，每个GNN层的激活（输出嵌入）通常必须存储起来，以便在反向传播 (backpropagation)期间进行梯度计算。对于一个平均维度为$d_{avg}$的$L$层GNN，这需要$O(L \times |V| \times d_{avg})$的内存。当$|V|$达到数百万或数十亿时，这会迅速耗尽即使高端GPU的内存容量（通常为数十到数百千兆字节）。

考虑一个具有1亿个节点、平均嵌入维度为128、2层且使用32位浮点数（4字节）的图。仅存储激活就需要大约$2 \times 100,000,000 \times 128 \times 4$字节，约为102.4 GB。这超出了大多数单个GPU的容量，需要专门的硬件或分布式设置，这会显著增加复杂性和成本。

### 邻居扩张问题

"多层GNN固有的“邻居扩张”问题加剧了这些计算和内存挑战。为了计算层$L$上单个节点的嵌入 (embedding)，GNN需要访问其$L$跳邻居的特征和结构。在许多图（特别是那些具有中心节点或小世界特性的图）中，这个$L$跳邻居的大小会随$L$呈指数级增长，即使是中等深度，也可能覆盖整个图的很大一部分。在全图方法下，训练期间为每个节点加载和处理这些可能极其庞大的邻居是不切实际的。"

这些可扩展性挑战需要我们放弃简单的全图训练方法。标准深度学习 (deep learning)技术如小批量随机梯度下降 (gradient descent)（SGD）是有效的，因为批次可以独立处理。然而，简单地独立采样节点会破坏GNN的图结构。节点之间通过边的依赖关系意味着一个节点的计算依赖于其邻居，而这些邻居可能不会被包含在一个简单的逐节点小批量中。

因此，需要专门的技术来实现大型图上的GNN训练。这些方法旨在将计算和内存需求分解为可管理的部分，同时仍然有效地运用图结构。接下来的章节将介绍旨在解决这些可扩展性限制的主要策略，包括邻居采样、图采样和基于子图的训练方法。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了基础的图卷积网络（GCN）模型，阐述了导致可伸缩性挑战的核心计算。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE，一种通过归纳学习和邻域采样解决大型图GNN可伸缩性的开创性方法。
