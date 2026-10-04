---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-tasks-techniques"
lesson: "heterogeneous-graphs-gnns"
sourceId: 2721
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/heterogeneous-graphs-gnns"
title: "处理异构图 (RGCN, HAN)"
description: "实现关系图卷积网络 (RGCN) 和异构注意力网络 (HAN) 等图神经网络模型，以应对具有多种节点/边类型的图。"
order: 1
plots: []
sourceHash: "c7d758f813750504447cbdb896cad2c3eb5f308a639ac3cc384f8fc92ab1185d"
sourceCorrections: []
---

图数据通常包含不同种类的实体和连接。考虑一个文献网络，其中有“作者”、“论文”和“会议”节点，通过“写作”、“引用”和“发表于”边连接。标准图神经网络 (neural network)主要为同构图（单一节点和边类型）设计，难以有效地捕捉这些多样化关系中蕴含的丰富含义。在所有边类型上应用单一的消息传递函数，会隐含地将它们视为相同，这通常过于简化。处理异构图需要专门设计的架构，以区分并运用各种节点和边类型的独特属性。

两种为异构图开发的著名图神经网络架构是：关系图卷积网络 (RGCN) 和异构注意力网络 (HAN)。

### 异构性的挑战

在同构图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中，标准的图神经网络 (neural network)层通过聚合来自其邻居 $\mathcal{N}(v)$ 的消息来更新节点 $v$ 的表示 $h_v$：


$$
h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{c_{uv}} W^{(l)} h_u^{(l)} \right)
$$


在这里，$W^{(l)}$ 是层 $l$ 的共享权重 (weight)矩阵，$c_{uv}$ 是一个归一化 (normalization)常数。异构图中的核心问题，例如 $\mathcal{G}' = (\mathcal{V}, \mathcal{E}, \mathcal{T}_V, \mathcal{T}_E)$，在此 $\mathcal{T}_V$ 和 $\mathcal{T}_E$ 分别是节点和边类型的集合，在于单一的 $W^{(l)}$ 无法充分模拟与不同关系类型 $r \in \mathcal{T}_E$ 相关的独特转换。例如，一个“作者”节点通过“写作”边影响“论文”节点的方式，与一个“论文”节点通过“引用”边影响另一个“论文”节点的方式，在语义上是不同的。

> 一个简单的异构图，呈现了不同的节点类型（作者、论文、会议）和边类型（写作、引用、发表于）。

### 关系图卷积网络 (RGCN)

关系图卷积网络通过引入关系特定的转换矩阵来直接处理异构性。对于每种关系类型 $r \in \mathcal{T}_E$，都会学习一个独特的权重 (weight)矩阵 $W_r^{(l)}$。在关系图卷积网络层中，节点 $v$ 的消息传递更新变为：


$$
h_v^{(l+1)} = \sigma \left( W_{self}^{(l)} h_v^{(l)} + \sum_{r \in \mathcal{T}_E} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{uv,r}} W_r^{(l)} h_u^{(l)} \right)
$$


在此，$\mathcal{N}_r(v)$ 是通过类型 $r$ 的边连接到节点 $v$ 的邻居集合，$W_r^{(l)}$ 是层 $l$ 上关系 $r$ 的权重矩阵，$W_{self}^{(l)}$ 是自连接的权重矩阵（可选但常见），且 $c_{uv,r}$ 是关系特定的归一化 (normalization)常数（例如 $|\mathcal{N}_r(v)|$）。

这种表述使模型能够根据连接两个节点的关系类型来学习不同的转换。

#### 参数 (parameter)化挑战与正则化 (regularization)

关系图卷积网络的一个重要实际问题是，当关系类型数量 $|\mathcal{T}_E|$ 很大时。为每种关系学习一个单独的密集矩阵 $W_r^{(l)}$ 会导致参数量巨大，增加过拟合 (overfitting)的风险，尤其是在每种关系类型的训练数据有限的情况下。

为此，关系图卷积网络常采用正则化方法：

1. **基分解**：关系特定的矩阵 $W_r^{(l)}$ 被限制为一小组共享基变换 $B_i^{(l)}$ 的线性组合：

   
   $$
   W_r^{(l)} = \sum_{i=1}^{B} a_{ri}^{(l)} B_i^{(l)}
   $$
   

   在这里，只需学习基矩阵 $B_i^{(l)}$ 和标量系数 $a_{ri}^{(l)}$，如果 $B \ll |\mathcal{T}_E|$，这将显著减少参数数量。
2. **块对角分解**：每个 $W_r^{(l)}$ 都被构造为块对角矩阵：

   
   $$
   W_r^{(l)} = \text{diag}(W_{r,1}^{(l)}, W_{r,2}^{(l)}, ..., W_{r,B}^{(l)})
   $$
   

   每个 $W_{r,i}^{(l)}$ 都是一个较小的密集矩阵。这会在权重矩阵中产生稀疏性并减少参数，本质上是将特征空间分解为关系独立运作的低维子空间。

这些方法使得关系图卷积网络能够扩展到具有多种关系类型的图，同时保持模型的能力。关系图卷积网络特别适用于知识图谱中的节点分类和链接预测等任务。

### 异构注意力网络 (HAN)

异构注意力网络采用另一种方法，运用注意力机制 (attention mechanism)自动学习不同邻居和关系类型（或者更普遍地，元路径）的重要性。它采用分层注意力结构：

1. **节点级注意力**：对于特定的元路径 $\Phi$（连接节点类型的一系列关系类型，例如 作者 $\xrightarrow{\text{写作}}$ 论文 $\xrightarrow{\text{引用}}$ 论文），异构注意力网络首先学习通过该元路径可达的邻居的注意力权重 (weight)。给定目标节点 $v$，对于通过元路径 $\Phi$ 连接的每个邻居 $u$，都会计算一个注意力权重 $a_{vu}^{\Phi}$，这类似于图注意力网络 (GAT)，但特定于 $\Phi$。这使得模型能够在该元路径的背景下优先考虑更相关的邻居。节点特征经过转换（可能使用类型特定的矩阵），然后利用这些注意力权重进行聚合，以获得元路径特定的嵌入 (embedding) $z_v^{\Phi}$。

   
   $$
   z_v^{\Phi} = \sigma \left( \sum_{u \in \mathcal{N}_v^{\Phi}} a_{vu}^{\Phi} \cdot W_{\Phi} h_u \right)
   $$
   

   在此，$\mathcal{N}_v^{\Phi}$ 是通过元路径 $\Phi$ 可达的邻居，且 $W_{\Phi}$ 是与该元路径关联的转换矩阵。
2. **语义级注意力**：由于不同的元路径捕捉不同的语义方面，并且可能对给定任务和节点具有不同的相关性，异构注意力网络引入了第二层注意力。它为所考虑的每个元路径 $\Phi_1, \Phi_2, ..., \Phi_P$ 学习注意力权重 $w_{\Phi}$。这些权重反映了元路径所捕捉的每个语义视图的重要性。最终的节点嵌入 $h_v'$ 通过将元路径特定的嵌入 $z_v^{\Phi_i}$ 与语义注意力分数加权组合获得：

   
   $$
   h_v' = \sum_{i=1}^{P} w_{\Phi_i} \cdot z_v^{\Phi_i}
   $$
   

   语义注意力权重 $w_{\Phi_i}$ 通常基于元路径嵌入本身计算，常涉及对其进行投影并使用 softmax 归一化 (normalization)。

#### 元路径

元路径是节点和边类型的序列（例如，作者-论文-作者，论文-会议-论文），它们在异构图中定义复合关系。它们通常根据领域知识预定义，是异构注意力网络运作的核心，指导邻居采样和注意力计算。元路径的选择明显影响异构注意力网络的性能。

HAN的优势在于其能够自适应地加权来自元路径定义的不同结构和语义上下文 (context)的信息，使其在复杂关系模式重要的节点分类任务中表现有效。

### 关系图卷积网络与异构注意力网络对比

- **机制**：关系图卷积网络为每种关系类型使用显式学习的转换。异构注意力网络使用由预定义元路径引导的分层注意力。
- **参数 (parameter)**：如果不进行正则化 (regularization)，关系图卷积网络可能参数量很大。异构注意力网络的参数数量取决于元路径的数量和注意力机制 (attention mechanism)的复杂程度。
- **可解释性**：异构注意力网络的注意力权重 (weight)（节点级和语义级）可以提供关于哪些邻居和元路径对节点表示影响最大的信息。关系图卷积网络的可解释性更多在于分析学习到的 $W_r$ 矩阵或基组件。
- **灵活性**：关系图卷积网络直接对定义的边类型进行建模。异构注意力网络依赖于元路径，元路径可以捕捉更远距离的依赖关系，但需要仔细定义。
- **实现**：关系图卷积网络和异构注意力网络的设计思路都得到了诸如 PyTorch Geometric（例如 `RGCNConv`、基于类似异构性原理构建的 `HGTConv`）和 DGL（例如 `RelGraphConv`、异构图处理模块）等库的支持。

在关系图卷积网络和异构注意力网络之间进行选择通常取决于具体的图结构、任务的性质以及是否能方便地定义有意义的元路径。两者都在将图神经网络 (neural network)应用于更复杂、更贴近实际的图数据方面取得了重要进展。

## 参考资料

- [Modeling Relational Data with Graph Convolutional Networks](https://doi.org/10.1007/978-3-319-93417-4_38) — Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling (2018)
  Journal: European Semantic Web Conference (ESWC); Publisher: Springer, Cham; Volume: 10843; Pages: 593-607; DOI: [10.1007/978-3-319-93417-4_38](https://doi.org/10.1007/978-3-319-93417-4_38)
  提出了关系图卷积网络 (RGCN) 架构，用于知识图谱中的关系数据建模，并介绍了参数正则化技术。
- [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293) — Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, P. Yu, Yanfang Ye (2019)
  Journal: The World Wide Web Conference (WWW); Pages: 2707-2718; DOI: [10.48550/arXiv.1903.07293](https://doi.org/10.48550/arXiv.1903.07293)
  介绍了异构图注意力网络 (HAN)，该网络利用分层注意力机制在异构图中通过不同元路径聚合信息。
