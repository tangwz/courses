# NetworkX 库简介

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning/introduction-to-networkx)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在代码中构建和操作图对象需要专门的工具。对于 Python 生态系统，NetworkX 是为此设计的标准库。它提供了图的数据结构，以及大量的图算法。虽然 NetworkX 不是深度学习 (deep learning)库，但它在加载、预处理和分析图数据方面的作用使其成为 GNN 工具箱中不可或缺的一部分。

### NetworkX 的核心操作

使用 NetworkX 非常简单。最主要的对象是 `Graph`，它以灵活的方式存储节点和边。让我们创建一个简单的无向图。

首先，导入该库，通常别名为 `nx`：

```python
import networkx as nx
```

接着，可以实例化一个空的图对象。

```python
# 创建一个空的无向图
G = nx.Graph()
```

你可以单独添加节点和边，也可以从列表或元组中批量添加。

```python
# 添加单个节点
G.add_node(0)
G.add_node(1)

# 从列表中添加多个节点
G.add_nodes_from([2, 3])

# 在节点 0 和 1 之间添加一条边
G.add_edge(0, 1)

# 从元组列表中添加多条边
G.add_edges_from([(1, 2), (2, 3), (3, 0)])
```

图构建完成后，可以查看其基本属性：

```python
print(f"节点总数: {G.number_of_nodes()}")
# 输出: 节点总数: 4

print(f"边总数: {G.number_of_edges()}")
# 输出: 边总数: 4

# 列出所有节点
print(f"节点: {list(G.nodes)}")
# 输出: 节点: [0, 1, 2, 3]

# 查找特定节点的邻居
print(f"节点 1 的邻居: {list(G.neighbors(1))}")
# 输出: 节点 1 的邻居: [0, 2]
```

### 为节点和边附加数据

在机器学习 (machine learning)中，图不仅仅是连接关系。节点和边通常带有相关的特征。NetworkX 通过属性来处理这些信息，这些属性以 Python 字典的形式存储。这就是我们开始将图结构与 GNN 使用的特征矩阵 ($X$) 联系起来的方式。

你可以在创建节点时添加属性，也可以稍后更新它们。

```python
# 添加带有属性的节点
G.add_node(0, community=0, role='user')
G.add_node(1, community=0, role='user')
G.add_node(2, community=1, role='moderator')
G.add_node(3, community=1, role='user')

# 也可以为边添加属性，例如权重
G.add_edge(0, 1, weight=0.5)
G.add_edge(2, 3, weight=0.9)
```

要访问这些属性，可以将 `G.nodes` 和 `G.edges` 对象视为字典。

```python
# 访问特定节点的属性
print(G.nodes[2]['role'])
# 输出: moderator

# 访问特定边的属性
print(G.edges[0, 1]['weight'])
# 输出: 0.5
```

所有节点的这些属性集合构成了我们之前讨论的节点特征矩阵的基础。例如，如果每个节点都有一个特征向量 (vector)，你可以将其作为 NumPy 数组或列表存储在每个节点的 'features' 属性中。

> 使用 NetworkX 创建的简单图。节点根据其 'community'（社区）属性着色，部分边带有相关的 'weight'（权重 (weight)）。

### 有向图和多重图

NetworkX 还支持其他类型的图。如果数据中的关系具有方向性（例如社交网络中的“关注”关系），则应使用 `DiGraph`（有向图）。

```python
# 创建一个有向图
DG = nx.DiGraph()

DG.add_edge('A', 'B') # 这创建了一条从 A 到 B 的边
DG.add_edge('B', 'C')

# 在有向图中，邻居默认是后继节点
print(list(DG.successors('A'))) # 输出: ['B']
print(list(DG.predecessors('A'))) # 输出: []

print(list(DG.successors('B'))) # 输出: ['C']
print(list(DG.predecessors('B'))) # 输出: ['A']
```

语法基本保持不变，但底层行为会遵循边的方向。

### NetworkX 在 GNN 工作流中的角色

NetworkX 非常适合图的操作、可视化以及应用传统的图算法（如中心性分析或社区发现）。它提供了一种易于理解且灵活的方式来处理图数据。

然而，它并未针对训练神经网络 (neural network)所需的高性能数值计算进行优化。深度学习 (deep learning)框架是在张量（Tensor）上运行的，而不是 NetworkX 的 `Graph` 对象。因此，常见的流程是使用 NetworkX 加载和预处理图，然后将其结构信息（邻接矩阵）和特征信息（节点属性）转换为 PyTorch 或 TensorFlow 等库所使用的张量。

在下一节中，我们将运用这些技能加载一个著名的图数据集并检查其属性，为之后在 GNN 模型中使用它做好准备。

## 参考资料

- [NetworkX Documentation](https://networkx.org/documentation/stable/) — NetworkX Developers (2025)
  使用 NetworkX 库的主要资源，包含其 API、数据结构和算法。
- [NetworkX-A Library for Network Analysis in Python](http://conference.scipy.org/proceedings/SciPy2008/paper_2/) — Aric A. Hagberg, Daniel A. Schult, Pieter J. Swart (2008)
  Journal: Proceedings of the 7th Python in Science Conference (SciPy2008); Publisher: Proceedings of the 7th Python in Science Conference; Pages: 11-15; DOI: [10.25080/TCWV9851](https://doi.org/10.25080/TCWV9851)
  介绍 NetworkX 的原始出版物，阐述其图分析的设计与功能。
- [Graph Theory](https://www.springer.com/book/9783662536216) — Reinhard Diestel (2017)
  Publisher: Springer; DOI: [10.1007/978-3-662-53621-6](https://doi.org/10.1007/978-3-662-53621-6)
  关于图论的全面权威教科书，为图结构和算法提供严谨的理论背景。

---

[上一节](05-%E5%9B%BE%E7%9A%84%E5%B1%9E%E6%80%A7%E4%B8%8E%E5%BA%A6%E9%87%8F.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8A%A0%E8%BD%BD%E5%B9%B6%E6%A3%80%E6%9F%A5%E5%9B%BE%E6%95%B0%E6%8D%AE%E9%9B%86.md)
