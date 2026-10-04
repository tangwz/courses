---
course: "introduction-to-graph-neural-networks"
chapter: "foundational-gnn-architectures"
lesson: "graphsage-model"
sourceId: 7628
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/graphsage-model"
title: "GraphSAGE：邻域采样与聚合"
description: "介绍 GraphSAGE 架构，该架构通过使用通用聚合函数和邻域采样对 GCN 进行了改进。"
order: 3
plots: []
sourceHash: "fc17329e201ab188d5a1a7360e7a704dcf9ecd7e398a559bcd261ab02d595a32"
sourceCorrections: []
---

图卷积网络 (GCN) 已成为从图数据中学习表示的一种高效且有效的方法。尽管 GCN 非常有用，但它存在两个主要的限制。首先，这类网络本质上是**转导式 (transductive)** 的。GCN 模型为训练图中的每个节点学习一个特定的嵌入 (embedding)，无法轻松地为训练后新加入的节点生成嵌入。其次，在大规模图上的计算成本可能很高，特别是对于拥有大量邻居的节点，因为每一层都需要处理整个邻域。

GraphSAGE（全称 Graph **SA**mpling and A**g**gr**e**gating，即图采样与聚合）的设计初衷就是为了直接解决这些挑战。它引入了一个不仅可以扩展到大规模图，还支持**归纳式 (inductive)** 学习的框架，使模型能够泛化到完全未见的节点。

GraphSAGE 的创新体现在两个方面：

1. **邻域采样：** GraphSAGE 不再使用节点的全部邻域进行聚合，而是首先在每一层采样固定数量的邻居。
2. **通用聚合器：** 它将 GCN 中特定的、不可训练的聚合方式替换为各种可学习的聚合函数。

通过学习如何从采样的一组局部邻居中聚合信息，模型学习到的是一个生成嵌入的**函数**，而不仅仅是记住现有节点的嵌入。

### GraphSAGE 的过程

在模型的每一层中，GraphSAGE 对每个节点执行以下两个步骤：

1. **采样：** 从目标节点的局部邻域中，随机采样固定数量的邻居。这一过程会根据 GNN 的层数重复进行。对于一个两层模型，首先采样目标节点的邻居，然后针对这些被采样的每个邻居，再采样**它们**的邻居。这为每个节点构建了一个固定大小的计算树。
2. **聚合：** 使用特殊的聚合函数从采样的邻居中收集信息。聚合后的信息随后与目标节点来自上一层的自身表示相结合，并通过神经网络 (neural network)层产生其新的表示。

下图展示了两层 GraphSAGE 模型的采样过程。为了计算中心节点 A 的最终嵌入 (embedding)，模型首先采样其部分邻居 (B, C, D)。然后，对于这些邻居中的每一个，再从它们各自的邻域中采样（例如，B 采样了 E 和 F）。聚合过程则是由外向内进行的。

> 对于节点 A，计算依赖于其采样的邻居 {B, C, D}，而这些邻居又依赖于它们各自采样的邻居。无论节点 A 的实际度数是多少，这都会创建一个固定大小的计算图。

这种采样策略确保了每个节点的计算成本是恒定的，与其度数无关。即使一个节点有成千上万个邻居，我们也只处理一个小的、固定大小的样本，从而使算法具有很强的可扩展性。

### 通用聚合函数

与使用固定均值聚合的 GCN 不同，GraphSAGE 使用不同的函数从采样的邻居中聚合信息。聚合器的选择会显著影响模型表现。设 $h_u^{k-1}$ 表示上一层 $k-1$ 中邻居节点 $u$ 的特征向量 (vector)，$\mathcal{S}_{\mathcal{N}(v)}$ 表示目标节点 $v$ 的采样邻居集合。

GraphSAGE 的作者提出了三种主要的聚合函数：

#### 均值聚合器 (Mean Aggregator)

这是最简单的选项，与 GCN 聚合器非常相似。它计算所有采样邻居特征向量的逐元素均值。


$$
h_{\mathcal{S}_{\mathcal{N}(v)}}^k = \frac{1}{|\mathcal{S}_{\mathcal{N}(v)}|} \sum_{u \in \mathcal{S}_{\mathcal{N}(v)}} h_u^{k-1}
$$


此函数简单直接且计算效率高。

#### LSTM 聚合器 (LSTM Aggregator)

在这种聚合器中，邻居被视为一个序列。LSTM（长短期记忆网络 (LSTM)，一种循环神经网络 (neural network) (RNN)）负责处理这个序列。由于 LSTM 对输入顺序敏感，而图中邻居并没有自然顺序，因此在训练期间会使用邻居的随机排列。这种聚合器表达能力更强，但实现和训练也更复杂。

#### 池化聚合器 (Pooling Aggregator)

池化聚合器通常效果最好。每个邻居的特征向量首先通过各自独立的且相同的多层感知机 (MLP)。变换后，应用对称的逐元素池化操作（如最大池化或均值池化）来聚合信息。


$$
h_{\mathcal{S}_{\mathcal{N}(v)}}^k = \text{最大值}(\{\sigma(\mathbf{W}_{\text{池化}} h_u^{k-1} + b), \forall u \in \mathcal{S}_{\mathcal{N}(v)}\})
$$


这里，$\mathbf{W}_{\text{池化}}$ 是一个可学习的权重 (weight)矩阵，$\sigma$ 是非线性激活函数 (activation function)，`最大值` 指的是逐元素最大池化。使用可训练的 MLP 赋予了这种聚合器极强的表达能力。

### 更新步骤

在聚合邻居表示 $h_{\mathcal{S}_{\mathcal{N}(v)}}^k$ 之后，GraphSAGE 将该向量 (vector)与目标节点来自上一层的自身表示 $h_v^{k-1}$ 相结合。与 GCN 的一个主要区别是，这两个向量在通过线性层和非线性激活函数 (activation function)之前先进行拼接 (concatenate)。


$$
h_v^k = \sigma \left( \mathbf{W}^k \cdot \text{拼接}(h_v^{k-1}, h_{\mathcal{S}_{\mathcal{N}(v)}}^k) \right)
$$


拼接操作明确保留了节点的上一个状态的表示，类似于残差网络中的“跳跃连接”。事实证明，这可以提升性能。矩阵 $\mathbf{W}^k$ 包含了第 $k$ 层的可学习权重 (weight)。

通过学习聚合器的权重（在使用池化聚合器的情况下）和更新函数，GraphSAGE 学习到了一个基于局部邻域结构生成节点嵌入 (embedding)的通用函数。这就是其具备归纳能力的原因，我们将在下一节中详细说明。

## 参考资料

- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍GraphSAGE的开创性论文，详细阐述了其归纳学习能力、邻域采样和广义聚合函数。
- [Graph Representation Learning](https://www.morganclaypool.com/doi/abs/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  由GraphSAGE作者之一撰写的综合性书籍，提供了图表示学习的深入内容，包括GNNs、归纳学习和可扩展性。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍图卷积网络的奠基性论文，为GraphSAGE旨在克服的转导性和可扩展性挑战提供了背景。
- [Inductive Learning: GraphSAGE (Stanford CS224W Lecture 7)](https://snap.stanford.edu/class/cs224w-2023/) — Jure Leskovec (2023)
  这份来自知名大学课程的讲义提供了GraphSAGE架构及其归纳能力易于理解的解释和可视化。
