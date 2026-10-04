---
course: "introduction-to-graph-neural-networks"
chapter: "gnn-implementation-pytorch-geometric"
lesson: "mini-batching-large-graphs"
sourceId: 7645
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/mini-batching-large-graphs"
title: "针对大规模图数据的分批处理"
description: "了解 PyTorch Geometric 如何处理多个图的分批操作，以便在大数据集上进行高效训练。"
order: 5
plots: []
sourceHash: "f690b293a5e74ec1ff9b89feccb703807d2a186c42666bb9b0e01532b9936074"
sourceCorrections: []
---

在使用标准的深度学习 (deep learning)模型（如处理图像的 CNN）时，数据的分批处理非常直观。你可以将多张固定大小的图像堆叠成一个张量。然而，图数据并不统一，每个图的节点和边数都不尽相同，因此无法将它们直接堆叠成整齐的矩形张量。一种常见的直观想法是将邻接矩阵填充到最大尺寸，但这会导致张量极其稀疏且占用大量内存，效率极低。

PyTorch Geometric 采用了一种高效的策略来应对这一挑战：它将一组小图组成的一个批次（mini-batch）视为一个由不连通组件构成的大图。这种方法并不是强行将图放入死板的张量结构中，而是将它们合并成一个可以在单次前向传播中处理的大型图。

### 不连通图的分批策略

核心思想是将一列小图组合成一个大的 `Data` 对象。假设我们有一个包含 $N$ 个图的批次。PyG 会创建一个包含这 $N$ 个图中所有节点和边的单一大图。由于各个原始图之间没有边相连，它们在大图结构中仍然保持为独立的组件。这使得 GNN 层能够正确执行消息传递，因为信息只会在每个原始子图的范围内流动。

这个过程主要分为三个步骤：

1. **拼接节点特征**：将批次中所有图的节点特征矩阵沿着节点维度进行拼接。
2. **拼接并偏移边索引**：边索引张量也会进行拼接。但是，后续每个图的索引必须增加前面所有图的累计节点总数。这能确保边能够指向新大特征矩阵中正确的节点。
3. **创建 `batch` 属性**：创建一个新的属性 `batch`。这是一个列向量 (vector)，用于将每个节点映射到它在批次中所属的原始图。例如，第一个图的所有节点索引为 0，第二个图的所有节点索引为 1，依此类推。

下图展示了如何将两个小图合并成一个分批后的图。

> 在这个分批后的图中，图 2 的节点索引偏移了 3（图 1 的节点数）。消息传递操作自然会限制在各自的原始子图中，因为蓝色节点组和绿色节点组之间不存在边。

### 使用 `DataLoader` 进行实践

幸运的是，你不需要手动执行这些分批操作。PyTorch Geometric 提供了一个 `DataLoader` 类，类似于标准 PyTorch 中的同名类，它可以自动处理整个过程。

你只需要将 `Data` 对象列表（即你的数据集）传递给 `DataLoader`，它就会产出代表整个批次的 `Data` 对象。

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# 加载一个包含许多小图的数据集
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# 创建 DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍历批次
for batch in loader:
    # batch 是一个代表 32 个图的单一 Data 对象
    print(batch)
    # > DataBatch(edge_index=[2, 2166], x=[597, 21], y=[32], batch=[597], ptr=[33])

    # batch.num_graphs 提供批次中的图数量
    print(f"批次中的图数量: {batch.num_graphs}")
    # > 批次中的图数量: 32
```

产出的 `DataBatch` 对象中的 `batch` 属性对于执行图级操作非常有用。例如，在图分类任务中，你会使用这个 `batch` 向量 (vector)来执行池化操作（如 `global_mean_pool`），从而分别为批次中的每个图聚合节点嵌入 (embedding)。

### 关于超大图采样的说明

这里描述的分批策略适用于包含许多中小型图的数据集，例如分子集合。它不适用于像社交网络这样的单一海量图。由于内存限制，在单次计算中处理拥有数百万个节点的图往往是不可行的。

对于单一的大图环境，会使用一种称为**邻域采样**的技术。这种方法不再对整个图进行分批，而是通过为一组根节点采样固定数量的邻居来创建小批次。这是 GraphSAGE 等模型采用的方法，使其能够在内存无法容纳的超大图上进行训练。PyTorch Geometric 也为此提供了工具，例如 `NeighborLoader`，不过这属于进阶内容，超出了初学入门的范围。

## 参考资料

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — PyTorch Geometric Developers (2024)
  官方文档详细介绍了 `torch_geometric.data.Data`、`torch_geometric.data.Batch` 和 `torch_geometric.loader.DataLoader`，这些是PyG中高效图批处理的核心。
- [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) — Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinícius Flores Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu (2018)
  Journal: arXiv:1806.01261 [cs.LG]; DOI: [10.48550/arXiv.1806.01261](https://doi.org/10.48550/arXiv.1806.01261)
  这篇综述论文提出了图神经网络及其计算操作的统一概念框架，提供了消息传递如何应用的理解，这对理解批处理策略为何有效非常有帮助。
- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) — Jure Leskovec, Charilaos Kanatsoulis (2025)
  Publisher: Stanford University
  一门知名的图神经网络大学课程，包括数据结构、批处理和各种GNN模型的实现细节。
