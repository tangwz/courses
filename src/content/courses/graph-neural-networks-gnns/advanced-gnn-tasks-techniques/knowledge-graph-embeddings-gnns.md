---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-tasks-techniques"
lesson: "knowledge-graph-embeddings-gnns"
sourceId: 2727
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/knowledge-graph-embeddings-gnns"
title: "结合GNN的知识图谱嵌入"
description: "运用GNN架构从知识图谱中学习嵌入，以完成链接预测等任务。"
order: 3
plots: []
sourceHash: "4e0f67b2f9af7156b348683fe46f3364f4419114887c82212c90795efcb9e800"
sourceCorrections: []
---

知识图谱（KG）以相互关联的实体和关系形式存储事实信息，常构成大型且复杂的图结构。例子有维基数据、Freebase，或生物学、金融等专业图。这些图常常是异构的，包含多种节点（实体）和边（关系）类型，这对其分析提出了具体的挑战。对这类结构化知识进行表示和推理 (inference)，对问答、推荐系统和数据整合等任务而言非常重要。

知识图谱嵌入 (embedding)（KGE）技术旨在为知识图谱中的实体和关系学习低维向量 (vector)表示（嵌入）。传统方法如TransE、DistMult和ComplEx，主要侧重于对单个三元组（头实体、关系、尾实体）内的关系进行建模，常表示为 $(h, r, t)$。尽管这些方法有效，但它们通常独立处理三元组，可能无法完整表达更宽广的图结构或涉及多跳的复杂关系模式。

图神经网络 (neural network)通过直接借助知识图谱的图结构，提供了一种强大的替代方案。GNN不再孤立地处理三元组，而是通过聚合其局部邻域的信息，并考虑连接它们的特定关系，来学习实体表示。

### GNN在知识图谱中的应用

主要思路是将实体视为节点，关系则可能视为GNN框架中的边类型或转换。这自然使得GNN能够将信息在图上传播，获取多跳关系路径以及实体间的结构相似性。

#### 关系图卷积网络（R-GCN）

关系图卷积网络（R-GCN）是专为知识图谱设计的最知名的GNN架构之一。R-GCN调整了GCN框架，以处理知识图谱固有的异构性，特别是多种关系类型。

在标准GCN中，消息聚合通常使用一个共享的权重 (weight)矩阵。而在知识图谱中，邻居的意义很大程度上取决于连接它的*关系*。R-GCN通过引入关系特定的转换来处理这个问题。节点（实体）$u$ 在第 $l+1$ 层的消息传递更新可以表示为：


$$
h_{u}^{(l+1)} = \sigma \left( \sum_{r \in \mathcal{R}} \sum_{v \in \mathcal{N}_{r}(u)} \frac{1}{c_{u,r}} W_{r}^{(l)} h_{v}^{(l)} + W_{0}^{(l)} h_{u}^{(l)} \right)
$$


这里：

- $h_u^{(l)}$ 是实体 $u$ 在第 $l$ 层的隐藏表示。
- $\mathcal{R}$ 是知识图谱中所有关系类型的集合。
- $\mathcal{N}_{r}(u)$ 是节点 $u$ 在关系 $r$ 下的邻居集合。
- $W_{r}^{(l)}$ 是第 $l$ 层中关系类型 $r$ 的可学习权重矩阵。这使得模型能够根据关系学习不同的转换。
- $W_{0}^{(l)}$ 是自连接（或自循环）的可学习权重矩阵，这使节点能够保留其前一层表示的信息。
- $c_{u,r}$ 是一个归一化 (normalization)常数，通常与节点 $u$ 在关系 $r$ 下的度有关，用于防止缩放问题。
- $\sigma$ 是一个非线性激活函数 (activation function)（例如ReLU）。

R-GCN的主要改进点在于对每种关系类型使用不同的 $W_r^{(l)}$ 矩阵。这使得GNN能够学习关系特定的消息转换，获取知识图谱中关系的不同语义。为了管理可能大量存在的各种关系，R-GCN常采用基分解或块对角分解技术，以正则化 (regularization)并减少与关系矩阵相关的参数 (parameter)数量。

> 'Alice'实体R-GCN消息传递示意图。来自邻居（'OrgX'，'Bob'）的消息在聚合前，使用关系特定权重（'works\_at'对应 $W_{r1}$，'friend\_of'对应 $W_{r2}$）进行转换。其中也包含一个自连接转换（$W_0$）。

#### 针对知识图谱的其他GNN方法

其他GNN架构已针对知识图谱进行调整或开发：

- **CompGCN（组合式GCN）：** CompGCN不为每种关系学习单独的权重，而是共同学习关系和实体的嵌入 (embedding)。它在消息传递框架内使用实体和关系嵌入之间的组合操作（如减法或乘法），有望提供更好的参数效率，并表达复杂的关系构成。
- **注意力机制 (attention mechanism)（例如GAT改编）：** 类似于图注意力网络（GAT）如何学习对邻居贡献赋予不同权重，注意力机制可以引入到针对知识图谱的GNN中。这使得模型在更新实体表示时，能够动态地确定不同邻居实体和关系的权重，有望提升在重要性不同的关系上的表现。这与本章引言中提及的异构注意力网络（HAN）相关联，HAN就是为知识图谱这类异构图设计的。

### 将GNN嵌入 (embedding)用于知识图谱任务：链接预测

知识图谱嵌入（包括由GNN生成的嵌入）的主要下游任务是**链接预测**。目标是预测知识图谱中缺失的链接（三元组）。给定一个不完整的三元组，如 $(h, r, ?)$（预测尾实体）或 $(?, r, t)$（预测头实体），模型应识别出最有可能的实体来补全该三元组。

有了GNN生成的实体嵌入 ($h_u$) 和可能学习到的关系嵌入 ($h_r$)，便可使用评分函数 $f(h_h, h_r, h_t)$ 来衡量三元组 $(h, r, t)$ 的合理性。常用评分函数有：

- **DistMult：** $f(h, r, t) = h_h^T \text{diag}(h_r) h_t$
- **ConvE：** 在与尾实体嵌入匹配之前，它使用一个二维卷积层对组合的头实体和关系嵌入进行操作。
- **RotatE：** 它将关系建模为复数空间中的旋转： $f(h, r, t) = -\| h_h \circ h_r - h_t \|$，其中 $\circ$ 是元素级乘积，嵌入在 $\mathbb{C}^k$ 中。

在链接预测的训练过程中，GNN模型和评分函数参数 (parameter)一同进行优化。通常，这包括最大化已知正向三元组的得分，同时最小化损坏或负向三元组的得分（即头实体或尾实体被随机实体替换的情况）。

### 实现与扩展性

PyTorch Geometric (PyG) 和 Deep Graph Library (DGL) 等库为异构图提供了专门支持，包括R-GCN层的有效实现以及处理不同节点和边类型的机制。这极大地简化了知识图谱GNN模型的构建。

然而，知识图谱可能规模庞大，包含数百万实体和数十亿三元组。由于规模原因，直接应用GNN在计算上可能面临挑战。第3章讨论的技术，例如邻域采样（GraphSAGE风格）、图采样（GraphSAINT）或子图训练（Cluster-GCN），通常对于在大规模知识图谱上良好地训练GNN是必要的。在链接预测训练中，仔细考量负采样策略对于性能和效率也同样重要。

### 优势与考量

使用GNN进行知识图谱嵌入 (embedding)提供了一些优势：

- **结构考量：** 直接考虑单个三元组的图结构。
- **端到端学习：** 实体表示是为下游任务（例如链接预测）专门学习的。
- **异构性处理：** R-GCN等模型明确设计用于处理具有多种关系类型的图。

潜在考量包括：

- **扩展性：** 在大型知识图谱上训练需要专门的采样或分区策略。
- **过平滑：** 知识图谱上的深层GNN模型可能会出现过平滑问题，使得实体表示过于相似。第3章的技术（残差连接、跳跃知识）可能适用。
- **关系表示：** 关系的建模方式（特定矩阵、嵌入、组合）显著影响性能和参数 (parameter)效率。

总而言之，GNN提供了一个灵活且强大的框架，用于从知识图谱中学习富有表现力的表示。通过将图结构直接集成到嵌入过程中，特别是通过R-GCN等模型，它们可以获取复杂的关系模式，这对链接预测等任务十分重要，从而提升了基于知识的系统的能力。

## 参考资料

- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) — Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling (2017)
  Journal: Extended Semantic Web Conference (ESWC); Pages: 593-607; DOI: [10.1007/978-3-319-93417-4_29](https://doi.org/10.1007/978-3-319-93417-4_29)
  介绍了关系图卷积网络（R-GCN），用于知识图谱建模，详细阐述了关系特定变换和参数正则化技术。
- [Composition-based Multi-Relational Graph Convolutional Networks](https://arxiv.org/abs/1911.03082) — Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Partha Talukdar (2020)
  Journal: International Conference on Learning Representations (ICLR) 2020; DOI: [10.48550/arXiv.1911.03082](https://doi.org/10.48550/arXiv.1911.03082)
  提出了基于组合的多关系图卷积网络（CompGCN），利用组合运算符学习实体和关系嵌入。
- [Graph Neural Networks for Knowledge Graph Completion: An Overview](https://www.jair.org/index.php/jair/article/view/12177) — Chengjin Zhang, Fan Zhou, Pengyuan Zhang, Yuqing Xia, Cheng Shi, and Jianlong Tan (2020)
  Journal: Journal of Artificial Intelligence Research; Publisher: AI Access Foundation; Volume: 67; Pages: 593-631; DOI: [10.1613/jair.1.12177](https://doi.org/10.1613/jair.1.12177)
  全面概述了应用于知识图谱补全的图神经网络，讨论了各种架构和任务。
- [Heterogeneous Graph Attention Network](https://doi.org/10.1145/3308558.3313562) — Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, Philip S. Yu (2019)
  Journal: The Web Conference (WWW); Publisher: ACM; Pages: 2022-2032; DOI: [10.1145/3308558.3313562](https://doi.org/10.1145/3308558.3313562)
  提出了异构图注意力网络（HAN），该网络在异构图上融合注意力机制实现分层聚合。
