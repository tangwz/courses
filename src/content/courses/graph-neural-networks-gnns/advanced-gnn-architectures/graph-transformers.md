---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-architectures"
lesson: "graph-transformers"
sourceId: 2678
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/graph-transformers"
title: "图Transformer"
description: "探讨Transformer架构在图数据上的应用，包括图的位置编码。"
order: 4
plots: []
sourceHash: "57bff84480cde1e5f62b796b6d619cf14a7c310b47df4e3b1da51d0112517c60"
sourceCorrections: []
---

Transformer架构最初在自然语言处理（NLP）中取得非凡成就，随后在计算机视觉中也表现出色，这激发了人们将其核心机制——自注意力 (self-attention)——应用于图结构数据的兴趣。与序列数据（文本）或网格状数据（图像）不同，图具有不规则的结构，且缺少规范的节点排序。将Transformer适配到这一范畴带来了独特的挑战，但也提供了潜在的优势，特别是在捕获图中的远距离依赖关系方面。

标准Transformer处理序列时，每个元素的位置都是明确定义的。然而，图具有排列不变性（或对于节点级别任务具有等变性），这意味着数据表示中的节点排序不应影响输出。直接将标准Transformer应用于一组节点特征（将其视为无序集合或任意序列），将忽略图边中编码的重要关系信息。

图Transformer旨在将强大的自注意力机制 (attention mechanism)与图数据的独特属性相结合。核心思想是让图中的每个节点都关注所有其他节点（或一个策略性选择的子集），学习衡量其他节点特征的重要性以更新自身的表示。

### 适配图的自注意力 (self-attention)机制 (attention mechanism)

回顾标准缩放点积注意力：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


在图Transformer语境下，查询（$Q$）、键（$K$）和值（$V$）矩阵通常源自节点特征矩阵 $H \in \mathbb{R}^{N \times d}$，其中 $N$ 代表节点数量，$d$ 代表特征维度。对于节点 $i$，其查询向量 (vector) $q_i$ 会关注所有其他节点 $j$ 的键向量 $k_j$（可能包括其自身）。节点 $i$ 和节点 $j$ 之间的注意力分数决定了节点 $j$ 的值向量 $v_j$ 对节点 $i$ 更新后的表示贡献了多少。

与图注意力网络（GATs）不同，GATs通常只计算节点直接邻居（由邻接矩阵 $A$ 定义）的注意力分数，而“纯”图Transformer潜在地可以计算*所有*节点对的注意力。这种全局注意力允许单层中远距离节点之间直接传播信息，潜在地缓解了与依赖局部聚合的深度消息传递GNN相关的过平滑和过压缩问题。

> 注意力模式对比。局部注意力（左侧）考虑直接邻居，而全局注意力（右侧）潜在地允许一个节点（如 'f'）直接关注所有其他节点，包括远距离节点（如 'k'）。

### 整合图结构

一个重要的挑战是如何让Transformer架构感知图的拓扑结构。标准Transformer依赖位置编码 (positional encoding)来告知模型序列顺序。我们如何向图Transformer注入结构信息？出现了几种策略：

1. **结构/位置编码：** 类似于NLP中的位置编码，我们可以使用源自图结构的信息来增强节点特征。常见方法包括：

   - **拉普拉斯特征向量 (vector)：** 使用图拉普拉斯矩阵（或归一化 (normalization)拉普拉斯矩阵等变体）的特征向量作为位置特征。这些特征向量捕获与图划分和平滑性相关的全局结构信息。节点 $i$ 的编码可能是 $[v_1(i), v_2(i), ..., v_k(i)]$，其中 $v_j$ 是第 $j$ 小的特征向量。
   - **随机游走概率：** 基于从节点开始的随机游走到达概率的特征。这捕获了局部邻域结构。例如，使用 $k$ 步随机游走概率。
   - **最短路径距离：** 编码节点间的最短路径距离（SPD）。这可以作为偏差添加到注意力机制 (attention mechanism)中，如果需要，促使图中距离较近的节点彼此更强地关注。
2. **注意力偏差：** 修改注意力分数计算，以明确地包含结构关系。例如，节点 $i$ 和节点 $j$ 之间的注意力分数可以根据它们的图距离或边特征进行修改：

   
   $$
   \text{分数}(i, j) = \frac{(q_i W_Q) (k_j W_K)^T}{\sqrt{d_k}} + \text{偏差}(i, j)
   $$
   

   其中 $\text{偏差}(i, j)$ 可以是基于 $i$ 和 $j$ 之间最短路径距离的学习嵌入 (embedding)，或者如果可用，基于边特征的学习嵌入。
3. **混合架构：** 结合消息传递层和Transformer层。消息传递层可以高效捕获局部结构，而Transformer层则可以在优化后的节点表示上建模全局交互。

### 计算考量

全局自注意力 (self-attention)机制 (attention mechanism)的主要缺点是其计算复杂度。计算所有 $N$ 个节点对之间的注意力分数每层需要 $O(N^2)$ 的计算和内存，这对于大型图而言是不可接受的。标准Transformer通常最多处理几千个标记 (token)的序列，而图可以轻松包含数百万或数十亿个节点。

为了解决这个问题，采用了几种技术：

- **稀疏注意力：** 将注意力机制限制在节点子集上，例如，仅限于 k 跳邻居或通过图稀疏化技术选择的节点。
- **高效近似：** 采用核化或低秩近似（例如 Linformer、Performer）等方法，这些方法经过调整以适用于图设置，用于近似完整的注意力矩阵计算。
- **子图采样：** 将图Transformer架构应用于采样子图，类似于可扩展的GNN训练技术。

### 何时使用图Transformer？

图Transformer与传统消息传递GNN相比，提供一套不同的归纳偏置 (bias)。

- 它们可能在**远距离交互**很重要的任务中表现出色，例如分子属性预测中远距离原子间的交互可能很重要，或在社交网络中了解社区之间的影响。
- 它们可以有效地应用于**中小型图**，其中 $O(N^2)$ 复杂度是可控的，或应用于**密集或全连接**的图，其中消息传递可能过快地扩散信息。
- 它们的性能很大程度上取决于所选**结构编码**方法的有效性。

然而，对于由局部交互主导的任务或非常大型的图，优化后的空间GNN（如高级GraphSAGE变体或PNA）或可扩展GNN方法可能提供更好的性能和效率权衡。在消息传递GNN、GAT或图Transformer之间选择取决于具体问题、图的特性和可用的计算资源。随着研究的进展，结合不同方法优势的混合模型也变得越来越普遍。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.55989/tq276h3k](https://doi.org/10.55989/tq276h3k)
  介绍了Transformer架构和缩放点积注意力机制，是图Transformer的基础。
- [Graph Attention Networks](https://openreview.net/forum?id=Bkg6kBMgZ) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  提出了用于图神经网络的局部注意力机制，有助于理解图中局部注意力与全局注意力之间的区别。
- [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/pdf/2106.05234) — Chengxuan Ying, Tianle Cai, Shengjie Zhong, Shuxin Zheng, Kaijie Xu, Wenzheng Feng, Mingxuan Wang, Bo Ren (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS; Volume: 34; Pages: 28877-28890; DOI: [10.48550/arXiv.2106.05234](https://doi.org/10.48550/arXiv.2106.05234)
  介绍了Graphormer，这是一种重要的图Transformer模型，它通过空间和中心性偏置等多种编码方式整合了图结构。
