# GAT 中的注意力机制

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/attention-mechanism-in-gat)

[返回章节目录](README.md) · [返回课程目录](../README.md)

某些图神经网络 (neural network)（GNN）架构（如 GCN 和 GraphSAGE）在处理节点的邻居时，认为它们的权重 (weight)是均等的，或者仅仅是由图结构决定的。例如，GCN 通过节点度数派生出的权重来对邻近节点的特征进行平均。这种方法计算效率很高，但存在明显的局限性。在许多图中，某些邻居的重要性远超其他邻居。以引用网络为例，一篇新文章引用的奠基性论文应当比一篇次要的参考文献拥有更高的权重。

图注意力网络（GAT）通过让每个节点学习其邻居的相对重要性来解决这个问题。GAT 不使用固定权重，而是为每条边计算注意力系数，从而在聚合步骤中学习节点对每个邻居应当投入多少“注意力”。这种机制受到了自然语言处理领域（尤其是 Transformer 模型）中成功的注意力模型的启发。

### 计算注意力系数

GAT 层的核心是自注意力 (self-attention)机制 (attention mechanism)，它为图中的每条边计算一个分数。这个分数即注意力系数 $e_{ij}$，它量化 (quantization)了节点 $j$ 的特征对节点 $i$ 的重要程度。

该过程始于两个相连节点 $i$ 和 $j$ 的输入特征 $h_i$ 和 $h_j$，其中 $h \in \mathbb{R}^{N \times F}$，$N$ 是节点数量，$F$ 是特征数量。

1. **线性变换：** 首先，对每个节点的特征向量 (vector)应用一个由权重 (weight)矩阵 $W$ 参数 (parameter)化的共享可学习线性变换。这会将特征投影到更高层的表示中。

   
   $$
   z_i = W h_i
   $$
   
2. **打分机制：** GAT 论文提出使用一个简单的单层前馈网络来计算注意力系数。将两个节点的变换后的特征向量 $z_i$ 和 $z_j$ 进行拼接。然后将这个组合向量乘以一个可学习的权重向量 $a$，随后应用 LeakyReLU 非线性激活函数 (activation function)。

   
   $$
   e_{ij} = \text{LeakyReLU}(a^T[z_i || z_j])
   $$
   

此计算会为从 $j$ 到 $i$ 的边生成一个标量分数 $e_{ij}$。这个分数是未经归一化 (normalization)的，代表了它们连接的原始重要性。对节点 $i$ 的每个邻居 $j$ 都会执行这一过程。

### 使用 Masked Softmax 进行归一化 (normalization)

这些原始注意力分数 $e_{ij}$ 不能直接使用，因为它们在不同节点之间难以比较。拥有许多邻居的节点的分数尺度可能与只有少数邻居的节点不同。为了使它们具有可比性并转化为概率分布，我们使用 softmax 函数对其进行归一化。

这里的一个细节是 **掩码注意力（Masked Attention）**。softmax 仅在节点 $i$ 的邻居集合 $\mathcal{N}_i$ 上应用。这旨在将图结构注入到机制中，确保节点仅与其直接邻居计算注意力分数。

最终的注意力权重 (weight) $\alpha_{ij}$ 计算如下：


$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}
$$


得到的权重 $\alpha_{ij}$ 为正值，且对于节点 $i$ 的所有邻居 $j$ 之和为 1。它们代表了节点 $i$ 分配给其邻居的注意力分布。

> 单个节点 `i` 及其邻居的 GAT 注意力机制 (attention mechanism)流程。该过程通过计算、归一化并应用注意力权重来创建更新后的节点表示。

### 应用注意力权重 (weight)

有了归一化 (normalization)后的注意力权重 $\alpha_{ij}$，最后一步是计算节点 $i$ 的新特征向量 (vector)（记作 $h'_i$）。这是通过对其所有邻居的线性变换特征进行加权求和来完成的。


$$
h'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} z_j\right) = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j\right)
$$


这里，$\sigma$ 表示应用于聚合特征的最终非线性激活函数 (activation function)，如 ReLU 或 ELU。整个过程构成了一个完整的 GAT 层。

### 通过多头注意力 (multi-head attention)提高稳定性

自注意力 (self-attention)的学习过程有时可能不稳定。为了缓解这一问题并使模型能够捕捉不同类型的关系，GAT 采用了**多头注意力（Multi-head Attention）**。这涉及并行运行多个独立的注意力机制 (attention mechanism)或“头”。

每个头 $k$ 都有自己的一组参数 (parameter)（$W^{(k)}$ 和 $a^{(k)}$）并计算自己的一组注意力权重 (weight) $\alpha_{ij}^{(k)}$。每个头生成一个独立的嵌入 (embedding)表示。然后将这些嵌入组合起来形成最终输出。通常有两种组合方式：

1. **拼接（用于中间层）：** 将所有 $K$ 个头的输出拼接在一起，创建一个更大的特征向量 (vector)。

   
   $$
   h'_i = \Big\|_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} W^{(k)} h_j\right)
   $$
   

   得到的输出维度为 $K \times F'$，其中 $F'$ 是单个头的输出维度。
2. **平均（用于最终层）：** 对于网络的最终输出层（例如用于分类），拼接并不合适。相反，会对来自不同头的嵌入进行平均。

   
   $$
   h'_{i, \text{最终}} = \sigma\left(\frac{1}{K}\sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} W^{(k)} h_j\right)
   $$
   

通过使用多头注意力，模型可以同时关注来自不同位置、不同表示子空间的信息。例如，一个头可能专注于社区结构，而另一个头则专注于局部连接模式，从而使整体模型更具表达力且更加稳定。

## 参考资料

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  引入图注意力网络并详细阐述其注意力机制和多头注意力原理的原始论文。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础性论文介绍了Transformer架构和自注意力机制，启发了GAT的注意力方法。

---

[上一节](05-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md) · [下一节](07-GCN%E3%80%81GraphSAGE%20%E4%B8%8E%20GAT%20%E7%9A%84%E6%AF%94%E8%BE%83.md)
