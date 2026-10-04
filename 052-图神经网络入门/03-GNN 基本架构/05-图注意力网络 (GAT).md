# 图注意力网络 (GAT)

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/graph-attention-networks-gat)

[返回章节目录](README.md) · [返回课程目录](../README.md)

图神经网络 (neural network)中的一个挑战是如何有效地聚合来自节点邻域的信息。许多现有的方法，如图卷积网络 (GCN) 和 GraphSAGE，以固定或统一的权重 (weight)来处理每个邻居。例如，GCN 层根据节点度分配权重，而使用均值聚合的 GraphSAGE 层则平等地对待每个邻居。这引发了一个问题：对于定义中心节点的作用或属性，所有的邻居都同样吗？

在许多图中，答案是否定的。例如，在引用网络中，来自一篇开创性论文的引用通常应该比来自一篇普通论文的引用具有更高的权重。图注意力网络 (GAT) 通过引入一种机制来解决这个问题，该机制允许模型学习不同邻居的相对权重。GAT 不使用固定的聚合权重，而是为每条边计算注意力系数，从而学习节点在聚合过程中应该对每个邻居投入多少“注意力”。

### 自注意力 (self-attention)机制 (attention mechanism)

GAT 层的核心是直接应用于图结构的自注意力机制。这个过程通过关注邻居来计算每个节点的更新特征。该操作可以分为几个不同的步骤。

首先，与其他 GNN 类似，对每个节点的特征向量 (vector) $h_i$ 应用一个由权重 (weight)矩阵 $W$ 参数 (parameter)化的共享线性变换。这将特征投影到一个可能不同的维度空间，使模型能更好地学习区分性属性。


$$
z_i = W h_i
$$


接着，对于从邻居 $j$ 到目标节点 $i$ 的每条边，模型会计算一个原始的、未归一化 (normalization)的注意力得分 $e_{ij}$。这个得分表示节点 $j$ 的特征对节点 $i$ 的权重。这通常由一个简单的单层前馈网络计算，该网络由权重向量 $a$ 参数化，并以两个节点的变换后特征向量拼接作为输入。


$$
e_{ij} = \text{LeakyReLU}(a^T [z_i || z_j]) = \text{LeakyReLU}(a^T [W h_i || W h_j])
$$


这里，$||$ 表示拼接。应用 LeakyReLU 激活函数 (activation function)是为了引入非线性。这种机制在图中所有的边上共享，这意味着模型学习的是一个计算注意力的通用函数。

这些原始得分 $e_{ij}$ 在不同的邻域之间不具备直接可比性。为了解决这个问题，我们对节点的所有邻居 $\mathcal{N}_i$ 使用 softmax 函数进行归一化。这会将原始得分转换为注意力系数 $\alpha_{ij}$ 的概率分布。


$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}
$$


生成的系数 $\alpha_{ij}$ 代表了学习到的邻居 $j$ 对节点 $i$ 的权重。

最后，节点 $i$ 的更新特征向量（记作 $h'_i$）被计算为邻居变换后特征的加权和，使用注意力系数作为权重。通常会对结果应用激活函数 $\sigma$（如 ReLU）。


$$
h'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} z_j\right) = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j\right)
$$


这整个过程构成了一个 GAT 层。通过学习注意力权重 $\alpha_{ij}$，模型可以动态调整每个邻居的影响，与 GCN 的静态聚合相比，这在表达能力上有了明显的提升。

> GAT 层为指向目标节点的每条边计算注意力系数 ($\alpha$)。这些系数决定了每个邻居对目标节点更新表示的贡献权重。在此图中，邻居 $j_2$ 具有最高的注意力权重。

### 多头注意力 (multi-head attention)

为了使学习过程更稳定，并允许模型捕捉不同类型的关系，GAT 采用了多头注意力机制 (attention mechanism)。这类似于卷积神经网络 (neural network) (CNN)使用多个卷积核来捕捉不同的特征（如垂直边缘、水平边缘、颜色）。

在多头注意力中，几个独立的注意力机制（或称“头”）并行执行注意力计算。每个头都有自己的一组参数 (parameter)（第 $k$ 个头的 $W^k$ 和 $a^k$），并计算自己的一组注意力系数 $\alpha_{ij}^k$。

每个头都会生成一个嵌入 (embedding)。然后将这些嵌入组合起来形成最终输出。对于中间层，通常将输出进行拼接。如果我们使用 $K$ 个注意力头，公式如下：


$$
h'_i = \Big\Vert_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j\right)
$$


其中 $||$ 再次表示拼接。这产生的输出特征向量 (vector)的大小是单个头输出的 $K$ 倍。

对于网络的最后一层，拼接不再合适。相反，通常在应用最终激活函数 (activation function)之前对各头的输出求平均值。


$$
h'_i = \sigma\left(\frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j\right)
$$


使用多头注意力有助于模型学习更丰富的特征，因为每个头可以专注于邻域结构和特征空间的不同方面。

### GAT 的特性

图注意力网络具有以下优点：

- **计算效率：** 自注意力 (self-attention)操作可以在所有边上并行执行，输出特征的计算可以在所有节点上并行执行。
- **归纳能力：** 与 GraphSAGE 类似，GAT 是归纳式的。注意力机制 (attention mechanism)在所有节点上共享，且仅依赖于局部邻域信息。这意味着在某个图上训练的 GAT 可以直接用于为另一个完全不同的图中的节点生成嵌入 (embedding)。
- **表达能力：** 通过允许节点为邻居分配不同的权重 (weight)，GAT 通常比 GCN 具有更强的表达能力，并能在具有复杂关系结构的任务中表现更好。

主要的权衡在于计算成本。与 GCN 或均值聚合的 GraphSAGE 中简单的聚合相比，为每条边计算注意力系数会增加开销，尤其是在非常稠密的图中。然而，性能的提升通常证明了这种额外成本是值得的。

## 参考资料

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: ICLR; DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  介绍图注意力网络的原始论文，详细阐述了其自注意力机制和多头注意力。
- [Deep Learning on Graphs](https://doi.org/10.1017/9781108924184) — Yao Ma and Jiliang Tang (2021)
  Publisher: Cambridge University Press; Pages: 400; DOI: [10.1017/9781108924184](https://doi.org/10.1017/9781108924184)
  一本关于各种图神经网络架构的书籍，在图上的深度学习的背景下，对GAT提供了结构化的理解。
- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) — Jure Leskovec (2025)
  Publisher: Stanford University
  斯坦福大学课程的讲义材料，提供了GAT及其他图神经网络的清晰解释和示例。

---

[上一节](04-GraphSAGE%20%E7%9A%84%E5%BD%92%E7%BA%B3%E5%AD%A6%E4%B9%A0.md) · [下一节](06-GAT%20%E4%B8%AD%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
