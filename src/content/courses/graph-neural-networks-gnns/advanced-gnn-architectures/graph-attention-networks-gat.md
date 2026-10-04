---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-architectures"
lesson: "graph-attention-networks-gat"
sourceId: 2673
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/graph-attention-networks-gat"
title: "图注意力网络 (GAT)"
description: "理解应用于图数据的自注意力机制，用于节点特征加权。"
order: 2
plots: []
sourceHash: "6828d3dfa9a829a15c1c7e004c54452340ecb812017a8280747355debc5818ef"
sourceCorrections: []
---

图卷积网络 (GCNs) 提供了一个在图上学习的强大框架，但它们的聚合机制通常对所有邻居一视同仁（仅根据节点度进行调整）。设想一个社交网络图，您想预测用户的兴趣。某些朋友可能比其他朋友影响力更大或兴趣更相似。标准 GCN 可能难以有效捕获这些差异，因为其卷积核权重 (weight)是根据图结构（特别是归一化 (normalization)邻接矩阵或拉普拉斯矩阵）固定的。

图注意力网络 (GATs) 引入了一种方法来克服这一局限，通过引入自注意力 (self-attention)机制 (attention mechanism)，这借鉴了它们在基于序列的任务中的成功。其核心理念是允许节点在聚合过程中，为其邻域内的不同节点赋予不同程度的重要性，即*注意力*。这些注意力权重并非固定，而是根据相互作用节点的特征动态计算。

### GAT 中的注意力机制 (attention mechanism)

GAT 不使用 GCN 中预定义的权重 (weight)（如 $1/\sqrt{deg(i)deg(j)}$），而是为每条边 $(j, i)$ 计算注意力系数 $a_{ij}$，这表示节点 $j$ 的特征对节点 $i$ 的重要性。这个过程通常在 GAT 层内包含三个步骤：

1. **线性变换：** 首先，将每个节点 $i$ 的输入特征 $h_i \in \mathbb{R}^F$ 通过一个可学习的权重矩阵 $W \in \mathbb{R}^{F' \times F}$ 进行变换，得到更高维度的特征 $z_i = W h_i \in \mathbb{R}^{F'}$。这种初始变换增加了模型学习相关表示的能力。
2. **注意力系数计算：** 对于连接节点 $j$ 到节点 $i$ 的每条边，都会计算一个*注意力得分* $e_{ij}$。这个得分表示节点 $j$ 的特征对节点 $i$ 未归一化 (normalization)的重要性。计算 $e_{ij}$ 的常用方法是使用共享注意力机制，这通常是一个由可学习权重向量 (vector) $a \in \mathbb{R}^{2F'}$ 参数 (parameter)化的单层前馈神经网络 (neural network)。该机制的输入是两个节点变换后特征 $z_i$ 和 $z_j$ 的拼接：

   
   $$
   e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j])
   $$
   

   这里，$||$ 表示拼接，LeakyReLU 通常用作非线性激活函数 (activation function)。这个机制在所有边之间是*共享*的，这意味着相同的 $a$ 和 $W$ 用于计算所有节点对的注意力得分。
3. **通过 Softmax 进行归一化：** 原始注意力得分 $e_{ij}$ 在不同节点之间不能直接比较。为了使它们可比并得到归一化的注意力系数 $a_{ij}$，Softmax 函数通常应用于节点 $i$ 的所有邻居 $j$（通常包括节点 $i$ 本身）：

   
   $$
   a_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i \cup \{i\}} \exp(e_{ik})}
   $$
   

   其中 $\mathcal{N}_i$ 表示节点 $i$ 的邻域。这些归一化系数 $a_{ij}$ 现在对于每个节点 $i$ 及其所有邻居 $j$ 求和为 1，形成了一个加权的注意力分布。
4. **加权聚合：** 最后，节点 $i$ 的输出特征 $h'_i \in \mathbb{R}^{F'}$ 是通过其邻居变换后特征的加权和计算得到的，其中使用归一化注意力系数 $a_{ij}$ 作为权重。之后通常会应用一个非线性激活函数 $\sigma$（如 ReLU 或 ELU）：

   
   $$
   h'_i = \sigma \left( \sum_{j \in \mathcal{N}_i \cup \{i\}} a_{ij} W h_j \right)
   $$
   

这个过程有效地使得每个节点在更新其表示时，能够有选择地侧重于其最相关的邻居。

> 节点 'i' 的 GAT 聚合过程视图。注意力系数 $a_{ij}$ 决定了从邻居（包括自身）传递过来的信息在更新节点特征向量 $h'_i$ 时的权重。

### GAT 的优势

图注意力网络与简单的卷积方法相比具有多个优势：

- **自适应感受野：** GAT 学习动态地分配重要性。节点学习*哪些*邻居应得到更多关注，从而根据节点特征而非仅仅图结构来调整聚合过程。
- **隐式加权：** 注意力系数 $a_{ij}$ 作为学习过程的一部分隐式计算，从而在正向传播过程中无需显式的图结构信息，例如度或代价高昂的矩阵运算（如拉普拉斯特征分解）。
- **效率：** 计算注意力系数涉及对边（节点对）的操作，可以高效地并行化，尤其是在稀疏图上。每层的计算不直接取决于图中的节点总数，这与某些谱方法不同。
- **归纳能力：** 由于注意力机制 (attention mechanism)依赖于节点特征并在所有节点之间共享，GAT 可以方便地应用于训练期间未见的图（归纳学习），前提是这些节点具有特征。

虽然单个 GAT 层计算一组注意力权重 (weight)，但这个机制可以扩展。一种常见做法是使用*多头注意力 (multi-head attention)*，其中多个独立的注意力机制并行计算，并将它们的结果组合。下一节将讨论这项技术，它通常能带来更稳定的训练，并使得模型能够同时捕获邻域内不同类型的关系。

## 参考资料

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  介绍图注意力网络（GAT）及其注意力机制的奠基性论文。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR) 2017; DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  关于图卷积网络（GCN）的原始论文，GAT在其基础上构建并解决了其局限性。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/8788931) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S. Yu Philip (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2970482](https://doi.org/10.1109/TNNLS.2020.2970482)
  一篇涵盖多种图神经网络架构（包括GAT）及其应用的广泛综述。
