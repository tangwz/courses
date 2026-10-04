---
course: "graph-neural-networks-gnns"
chapter: "gnn-foundations-revisited"
lesson: "graph-representations-ml"
sourceId: 2654
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-1-gnn-foundations-revisited/graph-representations-ml"
title: "机器学习中的图表示"
description: "考察图数据表示的高级方法，包括邻接矩阵、拉普拉斯矩阵和特征编码。"
order: 1
plots: []
sourceHash: "18949680b3c166a6e8e4bad09f553590b5850edb7ed8ab596d71552aa512bfab"
sourceCorrections: []
---

为了使用机器学习 (machine learning)模型，特别是图神经网络 (neural network)，有效处理图数据，我们首先需要将图的结构和相关信息转换为合适的数值格式。尽管第一章回顾了核心的消息传递思想，但图的特定*表示方式*很大程度上影响着这些消息的计算方式以及模型能学习到图的哪些属性。接下来，我们将考察图用于计算的标准和更高级的编码方式。

### 图结构表示

表示图 $G = (V, E)$ 连接性的最基本方式是**邻接矩阵**，此处的 $V$ 代表节点（顶点）集合，$E$ 代表边的集合。

#### 邻接矩阵 ($A$)

对于一个包含 $N = |V|$ 个节点的图，邻接矩阵 $A$ 是一个 $N \times N$ 矩阵，它满足：


$$
A_{ij} =
\begin{cases}
1 & \text{如果 } (v_i, v_j) \in E \\
0 & \text{否则}
\end{cases}
$$


对于加权图，$A_{ij}$ 可以表示节点 $v_i$ 和节点 $v_j$ 之间边的权重 (weight)。对于无向图，$A$ 是对称的（$A = A^T$）。

> 一个包含5个节点和5条边的简单无向图。

对于上图，邻接矩阵（假设节点索引从0到4）为：


$$
A = \begin{pmatrix}
0 & 1 & 1 & 0 & 0 \\
1 & 0 & 1 & 1 & 0 \\
1 & 1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 0
\end{pmatrix}
$$


"图通常是稀疏的，意味着 $A$ 中的大多数元素为零。这种稀疏性对于计算效率很重要。对于大型 $N$，存储和操作稠密的 $N \times N$ 矩阵变得非常耗时或不可行。因此，GNN库中通常使用稀疏矩阵格式（如坐标列表（COO）、压缩稀疏行（CSR）或压缩稀疏列（CSC））。我们将在第5章进一步讨论这些格式。邻接矩阵直接告知了消息传递框架中使用的邻居集合 $\mathcal{N}(v)$。"

#### 图拉普拉斯矩阵

邻接矩阵捕获直接连接，而图拉普拉斯矩阵则以一种对分析图属性和构成谱GNN依据特别有益的方式，编码了节点度数和图结构的信息。

首先，我们定义**度矩阵** $D$。这是一个 $N \times N$ 的对角矩阵，此矩阵中，$D_{ii}$ 是节点 $v_i$ 的度数（与其连接的边数），所有非对角元素为零。


$$
D_{ii} = \text{度}(v_i) = \sum_{j} A_{ij}
$$


使用 $A$ 和 $D$，我们可以定义几种形式的图拉普拉斯矩阵：

1. **组合拉普拉斯矩阵 ($L$)**:

   
   $$
   L = D - A
   $$
   

   这可能是最基本的形式。其属性与图连接性相关。例如，图中连通分量的数量等于 $L$ 的特征值0的重数。它是一个对称半正定矩阵。
2. **对称归一化 (normalization)拉普拉斯矩阵 ($\mathcal{L}_{sym}$)**:

   
   $$
   \mathcal{L}_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
   $$
   

   这里，$I$ 是单位矩阵，$D^{-1/2}$ 是对角矩阵，其元素为 $(D_{ii})^{-1/2}$。这种归一化使矩阵保持对称，并确保其特征值在0到2之间。这种边界限定在设计GNN的谱滤波器时具有优势，因为它可以防止数值不稳定。请注意，计算 $D^{-1/2}$ 要求 $D_{ii} > 0$。度数为0的节点可能需要根据应用进行特殊处理。
3. **随机游走归一化拉普拉斯矩阵 ($\mathcal{L}_{rw}$)**:

   
   $$
   \mathcal{L}_{rw} = D^{-1} L = I - D^{-1} A
   $$
   

   这个版本通常不对称。它与图上随机游走的转移矩阵（$P = D^{-1}A$）密切相关。$D^{-1}A$ 项表示邻居特征的平均，其中权重与发送节点的度数成反比，这构成了简单图卷积网络（GCN）中的核心聚合步骤。

拉普拉斯矩阵捕获关系信息的方式与邻接矩阵不同。它们的谱属性（特征值和特征向量 (vector)）展现了图的重要结构信息，例如连接性和划分，构成了谱图理论和谱GNN的依据，我们将在第1.3节和第2章进一步研究。

### 节点和边属性表示

机器学习 (machine learning)中的图很少只包含结构。节点和边通常带有属性或特征，提供有价值的信息。

#### 节点特征 ($X$)

节点特征通常表示为一个 $N \times F$ 矩阵 $X$，此矩阵中，$N$ 为节点数量，$F$ 为每个节点的特征数量。每一行 $X_v$ 对应于节点 $v$ 的特征向量 (vector)。


$$
X = \begin{pmatrix}
- & \mathbf{x}_1^T & - \\
- & \mathbf{x}_2^T & - \\
& \vdots & \\
- & \mathbf{x}_N^T & -
\end{pmatrix}
$$


这些特征可以是：

- **结构性：** 源自图本身（例如，节点度数、中心性度量、聚类系数）。
- **基于内容：** 与节点相关的外部信息（例如，社交网络中用户的文本资料、分子中原子的化学属性、图像中超像素的像素值）。
- **位置/时间性：** 表示节点在图中的位置或作用，或其在特定时间的状态的编码。

节点特征的性质和质量对GNN性能很重要。如果特征缺失，可以采用赋值常数、唯一的独热标识符或使用结构特征等方法。归一化 (normalization)或标准化等特征处理技术经常应用。初始节点特征矩阵 $X$ 通常作为消息传递框架中的输入 $\mathbf{h}_v^{(0)}$。

#### 边特征 ($E$)

边也可以拥有特征，特别是在知识图谱（边代表关系）、分子图（键类型）或交通网络（道路类型、旅行时间）等场景中。这些特征可以通过多种方式表示，通常是形状为 $|E| \times F_e$ 的张量 $E$，此处的 $F_e$ 是边特征的数量，或者根据连接两个节点的特定边直接集成到消息传递计算中。处理边特征需要特定的GNN架构，例如关系GCN (RGCN)，我们将在第4章介绍。

### 表示方式的考量

表示方式（$A$、$L$、$\mathcal{L}_{sym}$ 等）的选择以及特征（$X$、$E$）的处理直接影响：

- **计算效率：** 稀疏表示（COO/CSR 格式的 $A$）对大型图很重要。涉及稠密拉普拉斯矩阵的操作可能成为瓶颈。
- **模型设计：** 谱GNN明确使用拉普拉斯矩阵或其变体。空间GNN通常在其聚合步骤中隐含使用 $A$ 或归一化 (normalization)版本，如 $D^{-1/2} A D^{-1/2}$ 或 $D^{-1} A$。
- **捕获的信息：** 不同的表示方式侧重于图结构的不同方面。拉普拉斯矩阵编码谱方法中使用的平滑性和连接性信息。基本邻接矩阵编码空间消息传递中使用的直接邻居关系。
- **表达能力：** 正如我们将在第1.5节看到的，标准消息传递GNN（主要依赖于 $A$ 和 $X$）区分非同构图的能力有限，部分原因在于这些基本表示中编码信息的局部性质。需要更精巧的表示或GNN机制来捕获更高阶结构。

理解这些对图结构和特征进行数值编码的不同方式，是迈向构建和分析后续章节中讨论的先进GNN架构的第一步。我们现在有了词汇来以更高的精度讨论消息传递和谱方法。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）模型，展示了归一化邻接矩阵在GNN中实现有效消息传递和聚合的应用。
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/) — Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst (2021)
  Publisher: MIT Press
  为几何数据的深度学习提供了一个统一的数学框架，详细探讨了各种图表示及其在现代图神经网络中的作用。
- [Spectral Graph Theory (CBMS Regional Conference Series in Mathematics, No. 92)](https://bookstore.ams.org/cbms-92) — Fan R. K. Chung (1997)
  Publisher: American Mathematical Society; Volume: 92
  谱图理论的奠基性著作，深入探讨了图拉普拉斯算子及其性质，这对于理解谱GNN至关重要。
