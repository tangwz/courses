# PyG Data 对象

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/pyg-data-object)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了在一个统一且连贯的单元中管理图的各个组件（节点、边及其相关特征），PyTorch Geometric 引入了一种专门的结构：`Data` 对象。这个对象是 PyG 中所有图机器学习 (machine learning)任务的基础构建块，它将模型所需的所有图信息封装在一个便捷的容器中。

让我们来看看 `Data` 对象的主要属性，在构建 GNN 时，你会经常用到它们。

### Data 对象的核心属性

一个 `Data` 对象可以包含多个属性，但有几个属性对于定义图的结构和特征至关重要。所有这些属性都以 PyTorch 张量（tensor）的形式存储。

- **`data.x`**：节点特征矩阵。该张量存储图中每个节点的特征。它的形状为 `[num_nodes, num_node_features]`，其中 `num_nodes` 是节点总数，`num_node_features` 是每个节点特征向量 (vector)的维度。这与我们在前面章节中讨论的节点特征矩阵 `X` 相同。
- **`data.edge_index`**：图的连接性。这可能是最重要的属性。PyG 没有使用在处理稀疏图时效率较低的稠密邻接矩阵，而是采用了 **坐标格式 (COO)** 来表示图的连接。`edge_index` 是一个形状为 `[2, num_edges]` 且类型为 `torch.long` 的张量。第一行包含每条边的源节点索引，第二行包含相应的目标节点索引。这种表示方式对于实际应用中常见的稀疏图非常高效。
- **`data.edge_attr`**：边特征。在某些图中，连接本身也具有属性。例如，在分子图中，边可能代表不同类型的化学键。`edge_attr` 是一个可选的张量，形状为 `[num_edges, num_edge_features]`，用于存储这些特征。它的顺序必须与 `edge_index` 中的边顺序相对应。
- **`data.y`**：标签。这个可选属性存储用于训练模型的训练目标标签。`y` 的形状取决于具体任务。对于节点级任务（如节点分类），它的形状可能是 `[num_nodes]`。对于图级任务，它的形状通常为 `[1]`。

### 创建 Data 对象

让我们为一个简单的有向图构建一个 `Data` 对象，看看这些属性是如何工作的。考虑一个包含四个节点和四条有向边组成的环形图。

> 一个包含四个节点的有向图。连接关系由从节点 0 到 1、1 到 2、2 到 3 以及 3 回到 0 的边定义。

我们可以在 PyG 中如下表示这个图。假设每个节点都有一个 2 维特征向量 (vector)，并且我们有用于节点分类任务的标签。

```python
import torch
from torch_geometric.data import Data

# 定义图的连接性 (COO 格式)
# 边: 0->1, 1->2, 2->3, 3->0
edge_index = torch.tensor([
    [0, 1, 2, 3],  # 源节点
    [1, 2, 3, 0]   # 目标节点
], dtype=torch.long)

# 定义节点特征 (4 个节点，每个节点 2 个特征)
x = torch.tensor([
    [-1, 1],  # 节点 0 的特征
    [1, 1],   # 节点 1 的特征
    [1, -1],  # 节点 2 的特征
    [-1, -1]  # 节点 3 的特征
], dtype=torch.float)

# 定义节点标签
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# 创建 Data 对象
data = Data(x=x, edge_index=edge_index, y=y)

print(data)
```

运行这段代码会产生一个简洁汇总图内容的输出：

```
Data(x=[4, 2], edge_index=[2, 4], y=[4])
```

这个汇总让我们一目了然地看到图中有 4 个节点，每个节点有 2 个特征 (`x=[4, 2]`)，有 4 条边 (`edge_index=[2, 4]`)，以及 4 个对应的节点标签 (`y=[4]`)。

### 处理无向图

一个常见的问题是如何表示无向图，即节点 `u` 和 `v` 之间的边意味着双向连接。在 PyG 中，你必须显式地表示这一点。对于每条无向边，你需要在 `edge_index` 中添加两个条目：一个用于 `u -> v`，另一个用于 `v -> u`。

例如，如果我们的示例图是无向的，`edge_index` 将会是：

```python
# 对于无向图
undirected_edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],
    [1, 0, 2, 1, 3, 2, 0, 3]
], dtype=torch.long)
```

这种约定确保了在消息传递过程中，信息可以在连接的节点之间双向流动。

### 常用的属性和方法

`Data` 对象不仅仅是一个被动的数据容器，它还提供了一些实用的属性和方法来查看图的信息：

- **`data.num_nodes`**：返回图中的节点数（由 `x` 推断）。
- **`data.num_edges`**：返回图中的边数（由 `edge_index` 推断）。
- **`data.num_node_features`**：返回每个节点的特征数量。
- **`data.is_directed()`**：检查图是否有向。如果对于每条边 `(u, v)`，都同时存在边 `(v, u)`，则返回 `False`。
- **`data.is_undirected()`**：与 `is_directed()` 相反。

通过将图的整体结构和特征封装进一个定义良好的对象中，PyG 为构建模型提供了简洁且高效的基础。既然你已经了解了如何表示单个图，下一步就是通过 PyG 的 `Dataset` 类来学习如何处理图的集合。

## 参考资料

- [torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) — PyTorch Geometric Contributors (2024)
  PyTorch Geometric Data 对象的官方文档，详细说明其属性和方法。
- [PyTorch Geometric Documentation: Introduction to PyG](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) — PyTorch Geometric Contributors (2024)
  提供了 PyTorch Geometric 核心组件的概览，包括 Data 对象在该框架中的作用。

---

[上一节](01-PyTorch%20Geometric%20%28PyG%29%20%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20PyG%20%E6%95%B0%E6%8D%AE%E9%9B%86.md)
