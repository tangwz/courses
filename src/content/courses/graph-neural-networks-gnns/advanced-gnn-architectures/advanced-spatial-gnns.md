---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-architectures"
lesson: "advanced-spatial-gnns"
sourceId: 2683
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/advanced-spatial-gnns"
title: "进阶空间GNNs (GraphSAGE变体, PNA)"
description: "审视精巧的空间方法，涵盖GraphSAGE的改进与主成分邻域聚合 (PNA)。"
order: 6
plots: []
sourceHash: "4f427b7e9becd662aa73e7562da3998cd977b0d04dcf8effe00456f9ae5f1fb7"
sourceCorrections: []
---

基础空间GNNs，例如原始的GraphSAGE，通过聚合邻居信息，为从图结构中学习提供了有力的框架。但它们的效用有时会受限于所选聚合函数的类型。常见的聚合器，如均值、最大值或求和池化，各自存在固有的偏向。均值池化可能模糊掉来自特别有影响力的邻居的信息；最大值池化只关注最强的信号；而求和池化则可能受到节点度数的显著影响，从而可能导致不稳定或规模适应性问题。进阶空间GNNs通过引入更精密的聚合机制或结合多种策略来处理这些局限。

### GraphSAGE变体与改进

GraphSAGE框架本身具备灵活性，支持多种聚合函数。研究与实践已对最初的均值、最大池化和LSTM聚合器进行了变体研究：

1. **基于注意力的聚合：** 尽管图注意力网络（GAT）在前面已介绍，代表一种独特的架构，但在类似GraphSAGE的聚合步骤中引入注意力机制 (attention mechanism)，是一种普遍的改进。这使得模型能够动态学习不同邻居的重要性，在聚合时对它们的贡献进行加权，而非像均值那样统一对待，或仅依据特征强度（如最大值）。
2. **聚合器的结合：** 有些方法不选择单一聚合器，而是结合多个聚合器的输出（例如，拼接均值和最大池化的结果）。这使得模型可能同时捕获邻域分布的不同侧面。
3. **归一化 (normalization)技术：** 在聚合*之后*但在非线性激活和更新步骤*之前*应用层归一化或批归一化等技术，可以改善训练稳定性和性能，尤其是在更深层模型或邻域大小多样化的图中。

这些变体通常旨在提升聚合步骤的表达能力，使生成的节点嵌入 (embedding)更具信息量。然而，一种更具结构性的多聚合视角结合方法，促成了主成分邻域聚合的发展。

### 主成分邻域聚合 (PNA)

主成分邻域聚合 (PNA) 直接处理单一、简单聚合器的不足。其主要设想是，任何单一聚合器（均值、最大值、求和、标准差）都不足以获取节点邻域分布中包含的全部信息。PNA提出同时结合多个聚合器，并且，重要的一点是，明确地将度信息作为缩放因子引入。

#### 缘由

思考其局限：

- **均值/最大值聚合：** 它们对邻居数量（度数）不敏感。2个邻居的均值或最大值可能与200个邻居的结果相同，从而丧失可能重要的结构信息。
- **求和聚合：** 尽管对度数敏感，但对于高度数节点，其值可能变得过大；对于低度数节点，则可能趋于消失，使训练变得困难。它也未能很好地捕获特征的*分布*（例如方差）。

PNA旨在通过以下方式创建更全面的聚合函数：

1. 结合多个聚合器以捕获邻域的不同统计特性。
2. 使用基于度数的缩放器来调节聚合信息，使网络感知邻域大小。

#### PNA机制

在PNA层中，更新节点 $i$ 的表示 $h_i^{(k)}$ 至 $h_i^{(k+1)}$ 的聚合步骤涉及计算：


$$
h_i^{(k+1)} = U \left( h_i^{(k)}, \bigoplus_{\text{agg} \in A, s \in S} s(d_i) \cdot \text{agg}(\{ m_{ji}^{(k)} \mid j \in \mathcal{N}(i) \}) \right)
$$


此处：

- $h_i^{(k)}$ 是节点 $i$ 在第 $k$ 层的特征向量 (vector)。
- $m_{ji}^{(k)}$ 是从邻居 $j$ 到节点 $i$ 在第 $k$ 层的消息，通常派生自 $h_j^{(k)}$（例如，$m_{ji}^{(k)} = \text{MLP}(h_j^{(k)})$）。
- $A$ 是一组聚合函数。常见选择包括：
  - 均值 ($\mu$)：邻居消息的平均值。
  - 标准差 ($\sigma$)：邻居消息的离散程度。
  - 最大值 ($\max$)：按元素取最大值消息。
  - 最小值 ($\min$)：按元素取最小值消息。
- $d_i = |\mathcal{N}(i)|$ 是节点 $i$ 的入度。
- $S$ 是一组基于度数 $d_i$ 应用的缩放函数。典型缩放器包括：
  - 恒等：$s(d_i) = 1$。
  - 对数缩放器（增强）：$s(d_i) = \log(d_i + 1)$。增强高度数节点的信号。
  - 逆对数缩放器（衰减）：$s(d_i) = 1 / \log(d_i + 1)$。相对于低度数节点，衰减高度数节点的信号。
- $\bigoplus$ 代表一个组合运算符，通常是拼接操作，用于连接所有聚合器-缩放器对的结果。
- $U$ 是一个更新函数，通常是一个MLP，它将节点的先前表示与聚合和缩放后的邻域信息结合起来。

> 主成分邻域聚合 (PNA) 的流程。来自邻居的输入消息 $m_{ji}$ 由多个聚合器（均值、标准差、最大值、最小值等）处理。每个聚合结果随后使用节点度数 $d_i$ 的不同函数（恒等、对数、逆对数等）进行缩放。这些缩放后的结果被组合（例如，拼接），然后与节点的先前表示 $h_i^{(k)}$ 一同传递给更新函数 $U$，以生成新的表示 $h_i^{(k+1)}$。

#### 优势与考量

PNA在各种图机器学习 (machine learning)基准测试中常展现出强大的性能，尤其是在那些需要对细微结构差异敏感，或涉及节点度数差异很大的图中。通过明确结合邻域特征分布的多个统计矩并按度数进行缩放，PNA层能够生成比简单空间方法更丰富、更具区分性的节点表示。

实现PNA需要计算节点度数并应用多种聚合和缩放操作。PyTorch Geometric 等库提供了优化实现（例如，`PNAConv`）。每层由于多次聚合而增加的计算成本，是其增强表达能力所做的权衡。

### 比较进阶空间方法

- **GraphSAGE变体：** 通常比PNA更容易实现，计算量也更小。GraphSAGE中基于注意力的聚合可以提供动态加权，但它不像PNA那样，以相同的结构化方式明确地包含度数缩放或多个固定聚合器。
- **PNA：** 提供了一种规范的方法来结合多个固定聚合器和度信息。它在复杂图结构上可能表现更优，但每层会带来更高的计算开销。其效用取决于所选聚合器和缩放器集合是否适合任务和数据。

这些方法间的选择取决于具体问题、图数据的特性（特别是度分布）以及可用的计算资源。对于区分复杂局部结构重要的任务，PNA是一个有力的选项。对于计算成本是主要制约因素的超大型图，简单的GraphSAGE变体可能足够，甚至更可取，或者当它们与有效的采样策略（在第三章中讨论）结合使用时。

## 参考资料

- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了用于感应式节点嵌入生成的原始 GraphSAGE 框架，提出了平均、最大池化和 LSTM 等聚合函数，用于在大规模图上学习。
- [Principal Neighbourhood Aggregation for Graph Neural Networks](https://arxiv.org/abs/2004.05718) — Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2004.05718](https://doi.org/10.48550/arXiv.2004.05718)
  提出了主邻域聚合，这是一种 GNN 层，它结合了多个聚合函数和基于度数的缩放器，以捕获更丰富的邻域信息并有效处理不同节点度数。
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2017)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arxiv.1710.10903](https://doi.org/10.48550/arxiv.1710.10903)
  介绍了图注意力网络，这是一种神经网络架构，利用掩码自注意力层来学习邻居的加权贡献，影响了后续的基于注意力的 GNN 变体。
