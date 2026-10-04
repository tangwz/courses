---
course: "data-structures-algorithms-ml"
chapter: "graphs-for-ml"
lesson: "graph-representation"
sourceId: 5295
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-4-graphs-for-ml/graph-representation"
title: "图的表示方法：邻接列表与邻接矩阵"
description: "学习如何在内存中使用邻接列表和邻接矩阵表示图结构。"
order: 1
plots: []
sourceHash: "8fa1421e7b07eb7446bf7bae9768f32c66cca8a51e60b5958c75b0d7e965ce2c"
sourceCorrections: []
---

高效地在计算机内存中存储图的结构，对于将强大的图算法应用于机器学习 (machine learning)问题是不可或缺的。图 $G$ 由顶点（或节点）集 $V$ 和边集 $E$ 定义，其中每条边连接一对顶点。对 $V$ 和 $E$ 的表示方式的选择会很大程度上影响图算法在内存使用和计算时间方面的效率。

我们来看一个简单的示例图，以说明常见的表示方法。设想一个小型社交网络：

> 一个无向图，表示五个人（A、B、C、D、E）之间的连接。

我们将介绍表示此图（以及一般图）的两种主要方式：邻接矩阵和邻接列表。

### 邻接矩阵

邻接矩阵将图表示为一个方阵，通常记作 $M$。如果图有 $|V| = n$ 个顶点，则该矩阵的大小为 $n \times n$。顶点通常映射到从 $0$ 到 $n-1$ 的整数。

对于**无权图**，元素 $M[i][j]$ 为：

- 1，如果顶点 $i$ 和顶点 $j$ 之间存在边。
- 0，否则。

对于**带权图**，元素 $M[i][j]$ 为：

- 如果存在边，则是顶点 $i$ 和顶点 $j$ 之间边的权重 (weight)。
- 0 或 $\infty$（或其它表示无连接的指示符），如果没有直接边。如果边权重非零，则使用0是可行的。

对于**无向图**，邻接矩阵是对称的（$M[i][j] = M[j][i]$），因为 $i$ 和 $j$ 之间的边与 $j$ 和 $i$ 之间的边是相同的。对于**有向图**（边有方向，例如 $i \rightarrow j$），矩阵不一定对称。元素 $M[i][j]=1$ 表示存在一条从 $i$ 到 $j$ 的边。

**示例：社交网络图的邻接矩阵**

我们将 A 映射为 0，B 映射为 1，C 映射为 2，D 映射为 3，E 映射为 4。因为它是一个无向无权图：

```
   A B C D E
   (0 1 2 3 4)
A(0)[0 1 1 0 1]
B(1)[1 0 0 0 1]
C(2)[1 0 0 1 1]
D(3)[0 0 1 0 1]
E(4)[1 1 1 1 0]
```

在 Python 中，可以使用 NumPy 数组来表示：

```python
import numpy as np

# 示例图的邻接矩阵
adj_matrix = np.array([
  # A B C D E
    [0,1,1,0,1], # A
    [1,0,0,0,1], # B
    [1,0,0,1,1], # C
    [0,0,1,0,1], # D
    [1,1,1,1,0]  # E
])

# 检查C（索引2）和D（索引3）之间是否存在边
print(f"C和D之间有边吗？ {adj_matrix[2, 3] == 1}")

# 检查A（索引0）和D（索引3）之间是否存在边
print(f"A和D之间有边吗？ {adj_matrix[0, 3] == 1}")
```

**性能特点：**

- **空间复杂度：** 需要 $O(V^2)$ 空间，与边数无关。这对于*稀疏图*（边数相对于最大可能边数 $V^2$ 较少的图）来说效率很低。许多图，如社交网络或网络图，都是稀疏的。
- **时间复杂度：**
  - 检查顶点 $i$ 和 $j$ 之间是否存在边：$O(1)$（只需访问 $M[i][j]$）。
  - 查找顶点 $i$ 的所有邻居：$O(V)$（需要遍历整个行 $M[i]$）。
  - 添加或删除边：$O(1)$。
  - 添加顶点：需要调整矩阵大小，这通常是 $O(V^2)$。

### 邻接列表

邻接列表将图表示为列表（或类似的动态数组/集合）的集合。图中每个顶点都有一个列表。对于顶点 $u$，与 $u$ 关联的列表包含所有与 $u$ 有连接边的顶点 $v$。

- 对于**无权图**，顶点 $u$ 的列表仅包含其邻居的索引或标识符。
- 对于**带权图**，该列表通常为每个邻居 $v$ 存储对（或元组）：`(v, weight)`，表示邻居以及连接 $u$ 到 $v$ 的边的权重 (weight)。
- 对于**有向图**，顶点 $u$ 的列表仅包含与 $u$ 有边指向 $v$ 的顶点 $v$。

**示例：社交网络图的邻接列表**

在 Python 中使用字典，其中键是顶点，值是邻居列表：

```python
adj_list = {
    'A': ['B', 'C', 'E'],
    'B': ['A', 'E'],
    'C': ['A', 'D', 'E'],
    'D': ['C', 'E'],
    'E': ['A', 'B', 'C', 'D']
}

# 如果使用整数索引（0-4）：
adj_list_indexed = {
    0: [1, 2, 4], # A (0) 的邻居
    1: [0, 4],    # B (1) 的邻居
    2: [0, 3, 4], # C (2) 的邻居
    3: [2, 4],    # D (3) 的邻居
    4: [0, 1, 2, 3]  # E (4) 的邻居
}

# 查找C的邻居
print(f"C的邻居：{adj_list['C']}")

# 检查C和D之间是否存在边
print(f"C和D之间有边吗？ {'D' in adj_list['C']}")
```

**性能特点：**

- **空间复杂度：** 需要 $O(V + E)$ 空间。对于无向图，每条边 $(u, v)$ 出现两次（一次在 $u$ 的列表，一次在 $v$ 的列表），因此它与 $V$ 加上 $2E$ 成正比。对于有向图，每条边出现一次。这对于稀疏图来说非常节省内存，因为 $E$ 远小于 $V^2$。
- **时间复杂度：**
  - 检查顶点 $u$ 和 $v$ 之间是否存在边：$O(\text{degree}(u))$，其中 $\text{degree}(u)$ 是 $u$ 的邻居数量。我们需要扫描 $u$ 的列表以查看 $v$ 是否存在。使用集合而不是列表来存储邻居可以在平均情况下将其提高到 $O(1)$。
  - 查找顶点 $u$ 的所有邻居：$O(\text{degree}(u))$（只需遍历 $u$ 的列表）。这对稀疏图有效。
  - 添加边 $(u, v)$：平均 $O(1)$（假设列表追加是摊销常数时间）。
  - 添加顶点：平均 $O(1)$。

### 比较与权衡

邻接矩阵和邻接列表之间的选择在很大程度上取决于图的属性以及您需要频繁执行的操作。

| 特性 | 邻接矩阵 | 邻接列表 | 何时优先选择 |
| --- | --- | --- | --- |
| **空间** | $O(V^2)$ | $O(V+E)$ | **列表：** 稀疏图 ($E \ll V^2$) |
|  |  |  | **矩阵：** 稠密图 ($E \approx V^2$) |
| **检查边(u,v)** | $O(1)$ | $O(\text{degree}(u))$ / $O(1)^{*}$ | **矩阵：** 频繁的边检查 |
| **列出邻居(u)** | $O(V)$ | $O(\text{degree}(u))$ | **列表：** 频繁的邻居遍历 |
| **添加边** | $O(1)$ | $O(1)$ (平均) | (类似) |
| **删除边** | $O(1)$ | $O(\text{degree}(u))$ / $O(1)^{*}$ | **矩阵：** 频繁的边删除 |
| **添加顶点** | $O(V^2)$ (重设大小) | $O(1)$ (平均) | **列表：** 动态图大小 |

\*使用集合存储邻居使得邻接列表的检查/删除操作的平均时间复杂度为 $O(1)$。

在机器学习 (machine learning)环境中，表示现象（社交网络、用户-物品交互、分子结构）的图通常是大型且稀疏的。因此，**邻接列表通常是更推荐的表示方法**，因为它能显著节省空间并高效地遍历邻居，这是许多图算法（如接下来讨论的 BFS 和 DFS）的根本。邻接矩阵可以考虑用于较小、稠密的图，或者当极快的边存在性检查是首要任务时。

理解这些表示方法是实现和分析图算法性能的第一步，这对于构建推荐系统、分析网络结构以及支撑图神经网络 (neural network)等任务非常重要。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 22
  关于基本图表示及其属性的标准参考。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: Wiley; Pages: Chapter 14
  提供以 Python 为中心的数据结构方法，包括图表示的实际实现。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14; Pages: Chapter 2
  连接图表示与现代机器学习应用（特别是图神经网络）的资源。
