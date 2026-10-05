# 通用 GNN 层：聚合与更新

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/general-gnn-layer-aggregate-update)

[返回章节目录](README.md) · [返回课程目录](../README.md)

图神经网络 (neural network)的核心思想是学习一个函数，为每个节点生成新的特征向量 (vector)或“嵌入 (embedding)”（embedding）。这个新的嵌入是由节点自身的特征及其直接邻居的特征推导出来的。这一过程通常被称为“邻域聚合”，可以通过将单个 GNN 层分解为两个不同的步骤来形式化：\*\*聚合（AGGREGATE）**步骤和**更新（UPDATE）\*\*步骤。

这个两步过程是几乎所有现代 GNN 的基本计算模块。首先，一个节点收集来自其所有直接邻居的特征向量。然后，它利用这些聚合后的信息，结合其自身的当前特征向量，计算出下一层的新特征向量。

> 中心节点 $v$ 的两步消息传递过程示意图。来自邻居 $u_1, u_2, u_3$ 的信息首先在“聚合”步骤中结合，其结果随后在“更新”步骤中与节点自身的信息结合，从而生成新的表示 $v'$。

让我们更仔细地观察这些步骤。

### 聚合（AGGREGATE）函数

设计图神经网络 (neural network)的首要挑战是节点的邻居数量是可变的。社交网络中的一个节点可能有两名好友，也可能有两千名。标准的神经网络层（如全连接层）需要固定大小的输入向量 (vector)。我们该如何处理任意数量的邻居向量呢？

\*\*聚合（AGGREGATE）\*\*函数解决了这个问题。它的任务是接收一组邻居特征向量 $\{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \}$，并将它们合并为一个单一的、固定大小的向量。这个向量充当了节点整个邻域的摘要。


$$
\mathbf{m}_{\mathcal{N}(v)}^{(l)} = \text{聚合}^{(l)} \left( \{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \} \right)
$$


在这里，$\mathbf{m}_{\mathcal{N}(v)}^{(l)}$ 表示在第 $l$ 层从节点 $v$ 的邻域聚合而来的“消息”。

`聚合`函数的一个基本属性是它必须是\*\*置换不变性（permutation invariant）\*\*的。这意味着无论邻居向量出现的顺序如何，该函数都应产生相同的输出。简单的求和（sum）、平均值（mean）或最大值（max）运算是常见的选择，因为它们天然具备这种属性。我们将在下一节讨论这些选项。

### 更新（UPDATE）函数

一旦我们得到了代表邻域消息的单一向量 (vector)，\*\*更新（UPDATE）\*\*步骤就负责为下一层创建节点的新特征向量 $\mathbf{h}_v^{(l+1)}$。

该函数有两个输入：

1. 节点在当前层的特征向量 $\mathbf{h}_v^{(l)}$。
2. 来自`聚合`步骤的邻域聚合消息 $\mathbf{m}_{\mathcal{N}(v)}^{(l)}$。


$$
\mathbf{h}_v^{(l+1)} = \text{更新}^{(l)} \left( \mathbf{h}_v^{(l)}, \mathbf{m}_{\mathcal{N}(v)}^{(l)} \right)
$$


在这一步中包含节点自身的表示 $\mathbf{h}_v^{(l)}$ 非常有用。如果我们只使用聚合后的消息，节点就会丢失其原始信息，完全变成其周围环境的反映。通过结合现有状态和传入的消息，GNN 允许节点既保留自身特征，又整合来自局部图结构的信息。`更新`函数通常实现为标准的神经网络 (neural network)层，往往是一个线性变换后接一个非线性激活函数 (activation function)（如 ReLU）。

### 通用 GNN 层公式

通过结合这两个步骤，我们得到了消息传递 GNN 中单层计算的通用公式。该等式描述了任何节点 $v$ 的特征向量 (vector)在通过第 $l$ 层以产生第 $l+1$ 层输出时是如何变换的。


$$
\mathbf{h}_v^{(l+1)} = \text{更新}^{(l)} \left( \mathbf{h}_v^{(l)}, \text{聚合}^{(l)} \left( \{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)
$$


让我们拆解每个组成部分：

- $\mathbf{h}_v^{(l)}$ 是节点 $v$ 在当前第 $l$ 层的特征向量（或隐藏状态）。
- $\mathcal{N}(v)$ 是节点 $v$ 的所有邻居节点的集合。
- `聚合`函数（如 `mean` 或 `sum`）将所有邻居的特征向量 $\{ \mathbf{h}_u^{(l)} \}$ 压缩为一个向量。
- `更新`函数（通常是带有可学习参数 (parameter)的神经网络 (neural network)）将节点自身的向量 $\mathbf{h}_v^{(l)}$ 与聚合后的邻居向量相结合。
- $\mathbf{h}_v^{(l+1)}$ 是最终输出：节点 $v$ 在下一层 $l+1$ 的新特征向量。

对`聚合`和`更新`函数的具体选择，决定了 GCN、GraphSAGE 和 GAT 等不同 GNN 架构的区别。GNN 的可学习参数包含在这两个函数中。通过这一机制处理图中的所有节点，并堆叠多个此类层，GNN 就能从图结构中学习到复杂的模式。

## 参考资料

- [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf) — Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 70; Pages: 1263-1272; DOI: [10.5555/3305890.3306020](https://doi.org/10.5555/3305890.3306020)
  提出了通用的“消息传递神经网络”（MPNN）框架，将各种图神经网络变体的聚合-更新机制形式化。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  一篇基础论文，介绍了图卷积网络（GCN），该网络是广泛采用的GNN模型，体现了聚合-更新机制。
- [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5ff69caad792e626e6475658e3-Abstract.html) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; DOI: [10.5555/3295222.3295267](https://doi.org/10.5555/3295222.3295267)
  介绍了GraphSAGE，明确探讨了不同的邻域聚合函数（平均、LSTM、池化），并展示了消息传递框架的归纳能力。
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) — Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu (2018)
  Journal: arXiv preprint arXiv:1806.01261; DOI: [10.48550/arXiv.1806.01261](https://doi.org/10.48550/arXiv.1806.01261)
  提出了一个统一的“图网络”概念框架，通过关注对象、关系和全局属性，概括了包括消息传递机制在内的许多图神经网络方法。

---

[上一节](01-%E9%82%BB%E5%9F%9F%E8%81%9A%E5%90%88%E6%80%9D%E6%83%B3.md) · [下一节](03-%E5%B8%B8%E7%94%A8%E7%9A%84%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0.md)
