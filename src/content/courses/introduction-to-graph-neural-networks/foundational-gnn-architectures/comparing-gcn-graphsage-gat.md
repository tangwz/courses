---
course: "introduction-to-graph-neural-networks"
chapter: "foundational-gnn-architectures"
lesson: "comparing-gcn-graphsage-gat"
sourceId: 7632
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/comparing-gcn-graphsage-gat"
title: "GCN、GraphSAGE 与 GAT 的比较"
description: "GCN、GraphSAGE 和 GAT 架构的差异、优缺点摘要。"
order: 7
plots: []
sourceHash: "902fa1169d61d57931def68363d6aaf11c02f3653ff0b702be453b53139223ad"
sourceCorrections: []
---

直接对比 GCN、GraphSAGE 和 GAT 有助于理解它们各自的长处与短板。每种架构在消息传递框架中都做出了不同的设计选择，从而在性能、可扩展性和灵活性之间产生了取舍。选择合适的模型主要取决于具体问题、图的大小以及计算预算。

### 聚合机制

这些架构之间最根本的区别在于它们如何聚合来自节点邻域的消息。这一选择直接影响模型的表达能力和行为。

- **GCN** 使用一种简单且固定的聚合方案。它计算邻居特征向量 (vector)的归一化 (normalization)总和，这相当于加权平均值。权重 (weight)由图邻接矩阵中的节点度数决定，在训练过程中是不学习的。这使得 GCN 成为一种*各向同性 (isotropic)* 模型，意味着它对每个邻居都一视同仁。其更新规则计算效率高，但表达能力有限。
- **GraphSAGE** 通过引入灵活、可学习的聚合函数，使这一过程变得通用。你可以使用最大池化、均值，甚至像 LSTM 这样更复杂的神经网络 (neural network)来组合邻居消息，而不是使用固定的均值聚合器。这使得模型能够学习邻域内更复杂的关系。
- **GAT** 则采取了不同的方法，引入了注意力机制 (attention mechanism)。它为每个邻居计算注意力系数，有效地学习了对于给定任务，每个邻居对中心节点的影响力。随后，聚合过程变为加权求和，其中的权重就是这些学习到的注意力分数。这使得 GAT 成为一种*各向异性 (anisotropic)* 模型，因为它可以为同一邻域内的不同节点分配不同的权重，表现出极强的表达能力。

> 中心节点 `v` 及其邻居 `u` 的聚合机制。GCN 使用固定的均值，GraphSAGE 使用通用的聚合器 `AGG`，而 GAT 使用基于学习到的注意力系数 `α` 的加权和。

### 归纳式学习 vs. 直推式学习

实际应用中的一个主要区别在于模型是否能够进行归纳式学习。

- **直推式学习 (Transductive Learning)：** 模型在训练期间能看到整个图结构，包括所有节点（训练、验证和测试）的特征。它学习的是该图中特定节点的嵌入 (embedding)。GCN 本质上是直推式的，因为它的公式依赖于完整的图拉普拉斯矩阵，而该矩阵源自整个图的邻接矩阵。它很难为训练期间不在图中的节点生成嵌入。
- **归纳式学习 (Inductive Learning)：** 模型学习一个可以为任何节点生成嵌入的函数，包括从未见过的节点。它通过学习如何聚合节点局部邻域的特征来实现这一点，而无需考虑节点的身份。**GraphSAGE 和 GAT 都是归纳式的。** 它们学习的是聚合和特征转换函数的权重 (weight)，而不是特定节点的嵌入。这使得它们适用于新节点不断加入的动态图，或者将训练好的模型部署到完全不同的新图中。

### 计算复杂度与可扩展性

每个模型的计算需求各不相同，影响了它们处理大规模图的适用性。

- **GCN：** 训练 GCN 涉及与完整的归一化 (normalization)邻接矩阵进行稀疏矩阵乘法。虽然每一轮训练（epoch）速度很快，但模型需要将整个图结构存储在内存中，对于拥有数百万或数十亿节点的图来说，这是不可行的。
- **GraphSAGE：** 该架构专为可扩展性设计。通过在训练期间为每个节点采样固定大小的邻域，它避免了一次处理整个图。这使得每个批次（batch）的计算占用保持不变，不受图整体大小的影响。然而，采样过程会带来额外开销，在较小的图上，每轮训练的速度可能比 GCN 慢。
- **GAT：** 注意力机制 (attention mechanism)增加了计算成本。对于每个节点，GAT 必须计算其与所有邻居的注意力分数。如果一个节点有 $N$ 个邻居，特征维度为 $F$，则复杂度大致与 $N \times F$ 成正比。虽然这些计算是高度并行的，但比起 GCN 或 GraphSAGE 中的简单聚合，其开销更大，尤其是对于度数非常高的节点。

### 差异摘要

下表简要概括了这三种基础 GNN 架构的主要区别。

| 特性 | 图卷积网络 (GCN) | GraphSAGE | 图注意力网络 (GAT) |
| --- | --- | --- | --- |
| **聚合类型** | 固定均值（各向同性） | 灵活聚合器（均值、最大值、池化） | 学习得到的加权平均（各向异性） |
| **学习设置** | 直推式 | 归纳式 | 归纳式 |
| **可扩展性** | 受限于全图内存 | 高（通过邻域采样） | 中等（高度数节点计算成本高） |
| **表达能力** | 较低（平等对待所有邻居） | 中等（学习聚合函数） | 较高（学习每个邻居的重要性） |
| **主要优点** | 简单，对中等规模的静态图效率高。 | 可扩展至大规模图，具备归纳能力。 | 在邻居重要性差异大的任务中表现优异。 |
| **主要局限** | 非归纳式；需要在内存中存储全图。 | 采样增加了复杂性，每轮训练可能较慢。 | 每一层的计算成本较高。 |

### 应该选择哪种模型？

在 GCN、GraphSAGE 和 GAT 之间做出选择通常取决于几个实际问题：

1. **你的图是静态的且小到可以存入内存吗？** 如果是，**GCN** 是一个强力且简单的基准模型，通常很难被超越。它的高效率和直观的实现使其成为在 Cora 或 PubMed 等图上进行节点分类的绝佳起点。
2. **你需要泛化到未知节点或处理海量图吗？** 如果你的应用涉及动态图或内存无法承载的大规模图，**GraphSAGE** 是明确的首选。它的归纳特性和基于采样的训练正是为此类场景设计的。
3. **邻居的重要性是否变化很大且对任务至关重要？** 当你认为某些邻居比其他邻居重要得多时，**GAT** 是一个功能强大的选项。只要你有足够的计算资源进行有效训练，它的注意力机制 (attention mechanism)就能捕捉到这些关系，通常能达到顶尖的性能。例如蛋白质相互作用网络或关系不均一的知识图谱。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍图卷积网络（GCN）模型的原始论文，是理解同构消息传递的基础工作。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Neural Information Processing Systems (NIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE，一个使用可学习聚合函数和邻域采样进行归纳图表示学习的框架。
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  介绍了图注意力网络（GAT），它采用自注意力机制学习邻居的不同重要性权重，实现异构聚合。
