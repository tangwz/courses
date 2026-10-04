---
course: "advanced-python-programming-ml"
chapter: "advanced-data-structures-algorithms-ml"
lesson: "graph-data-structures-algorithms"
sourceId: 2327
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/graph-data-structures-algorithms"
title: "图数据结构与算法"
description: "在Python中表示和操作图数据；应用适用于机器学习任务的图算法（搜索、最短路径）。"
order: 2
plots: []
sourceHash: "186c6f54e7c3e63d921a18d74c402c0c8b9f1fe3094f4215a045c8766f2eb688"
sourceCorrections: []
---

图为实体间的关系和连接提供了强大的抽象，这在许多机器学习 (machine learning)问题中是一种常见的情况。与列表或树等线性或层级结构不同，图捕获了复杂的非线性关系，使其适合表示社交网络、推荐系统、分子结构、知识库和交通网络。了解如何在Python中高效表示和操作图数据对于应对此类机器学习任务很有帮助。

### 在Python中表示图

选择合适的图表示方法，很大程度上取决于图的属性（如密度、是否加权或有向）以及你计划频繁执行的操作。我们来看看常见方法：

#### 1. 邻接表

这通常是稀疏图（边的数量 $E$ 远小于可能的最大边数，即 $V$ 个顶点对应 $V^2$）最节省内存的表示方式。每个节点都关联一个与其相邻节点的列表（或集合）。在Python中，这通常使用字典实现，其中键是节点，值是邻居的列表或集合。

```python
# 示例：使用字典表示无向图邻接表
graph_adj_list = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

# 添加一条边（例如，D和F之间）
graph_adj_list.setdefault('D', set()).add('F')
graph_adj_list.setdefault('F', set()).add('D')

print(f"B的邻居：{graph_adj_list.get('B', set())}")
# 输出：B的邻居：{'E', 'D', 'A'}
```

对于加权图，列表可以存储元组 `(neighbor, weight)`。

```python
# 示例：加权有向图邻接表
graph_weighted_adj = {
    'A': {('B', 5), ('C', 3)},
    'B': {('D', 2), ('E', 4)},
    'C': {('F', 1)},
    'D': {},
    'E': {('F', 6)},
    'F': {}
}

# 获取边A -> B的权重
weight_A_B = None
if 'A' in graph_weighted_adj:
    for neighbor, weight in graph_weighted_adj['A']:
        if neighbor == 'B':
            weight_A_B = weight
            break
print(f"边A -> B的权重：{weight_A_B}")
# 输出：边A -> B的权重：5
```

**优点：** 对稀疏图内存效率高，易于遍历节点的邻居。
**缺点：** 检查特定边 $(u, v)$ 的存在性所需时间与节点 $u$ 的度数成正比。

#### 2. 邻接矩阵

邻接矩阵将图表示为一个 $V \times V$ 矩阵，通常使用NumPy数组。如果存在从节点 $i$ 到节点 $j$ 的边，则矩阵中的 `matrix[i][j]` 为1（或边的权重 (weight)），否则为0（或权重表示为无穷大/None）。对于无向图，矩阵是对称的。

```python
import numpy as np

# 假设节点映射到索引0..5（A=0，B=1，...，F=5）
nodes = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
num_nodes = len(nodes)

# 无权图示例的邻接矩阵
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('E', 'F')]

for u, v in edges:
    u_idx, v_idx = nodes[u], nodes[v]
    adj_matrix[u_idx, v_idx] = 1
    adj_matrix[v_idx, u_idx] = 1 # 对于无向图

print("邻接矩阵：")
print(adj_matrix)

# 检查边(B, D)是否存在
edge_exists = adj_matrix[nodes['B'], nodes['D']] == 1
print(f"\n边(B, D)存在：{edge_exists}")
# 输出：边(B, D)存在：True
```

**优点：** 快速检查边是否存在 $O(1)$。对稠密图高效。可以借助高度优化的矩阵操作（例如NumPy中）进行某些算法。
**缺点：** 需要 $O(V^2)$ 内存，对于大型稀疏图效率不高。遍历节点邻居需要 $O(V)$ 时间。

#### 3. 边列表

这简单地是一个元组列表，其中每个元组代表一条边 $(u, v)$ 或 $(u, v, weight)$。它直接明了，但除非处理成邻接表或邻接矩阵的形式，否则查找效率通常较低。

```python
# 示例：无权图的边列表
edge_list = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('E', 'F')]

print(f"边列表：{edge_list}")
```

### 使用库：NetworkX

对于大多数实际应用，你不会从头开始实现这些表示方法。像 `NetworkX` 这样的库提供了图对象和常用算法的实现。

```python
import networkx as nx
import matplotlib.pyplot as plt # 用于基本可视化

# 使用NetworkX创建图
G = nx.Graph() # 对于无向图

# 添加节点（可选，边缘添加时自动添加）
G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])

# 从边列表添加边
G.add_edges_from(edge_list)

# 添加之前添加的D-F边
G.add_edge('D', 'F')

print(f"\nNetworkX图节点：{list(G.nodes())}")
print(f"NetworkX图边：{list(G.edges())}")
print(f"B的邻居：{list(G.neighbors('B'))}")

# 基本可视化
nx.draw(G, with_labels=True, node_color='#a5d8ff', font_weight='bold')
plt.show()
```

`NetworkX` 内部主要使用邻接表表示，使其适用于多种图类型。

### 基本图算法

有几种算法在此类图结构上运行以提取有意义的信息。

#### 图遍历：BFS和DFS

遍历算法系统地访问图节点。

- **广度优先搜索 (BFS)：** 从源节点开始逐层搜索图。它使用队列来跟踪要访问的节点。BFS非常适合在无权图中查找按边数计算的最短路径。

  ```python
  from collections import deque

  def bfs_shortest_path(graph, start, goal):
      """使用BFS按边数查找最短路径。"""
      if start == goal:
          return [start]
      
      queue = deque([(start, [start])]) # (节点, 路径列表)
      visited = {start}

      while queue:
          current_node, path = queue.popleft()

          for neighbor in graph.get(current_node, set()):
              if neighbor == goal:
                  return path + [neighbor]
              if neighbor not in visited:
                  visited.add(neighbor)
                  queue.append((neighbor, path + [neighbor]))
      
      return None # 未找到路径

  # 使用之前的邻接表
  start_node = 'A'
  end_node = 'F'
  shortest_path = bfs_shortest_path(graph_adj_list, start_node, end_node)
  print(f"\n从{start_node}到{end_node}的最短路径：{shortest_path}")
  # 输出：从A到F的最短路径：['A', 'C', 'F'] 或 ['A', 'B', 'E', 'F'] 取决于顺序
  ```
- **深度优先搜索 (DFS)：** 在回溯之前，沿着每条分支尽可能深入。它通常使用递归或显式栈。DFS有助于检测环、拓扑排序（针对有向无环图 - DAG）以及查找连通分量。

  ```python
  def dfs_path_exists(graph, start, goal, visited=None):
      """使用DFS（递归）检查路径是否存在。"""
      if visited is None:
          visited = set()
      
      visited.add(start)
      if start == goal:
          return True
          
      for neighbor in graph.get(start, set()):
          if neighbor not in visited:
              if dfs_path_exists(graph, neighbor, goal, visited):
                  return True
      return False

  path_exists = dfs_path_exists(graph_adj_list, 'A', 'D')
  print(f"A和D之间存在路径：{path_exists}")
  # 输出：A和D之间存在路径：True

  path_exists_isolated = dfs_path_exists(graph_adj_list, 'A', 'G') # G不在图中
  print(f"A和G之间存在路径：{path_exists_isolated}")
  # 输出：A和G之间存在路径：False
  ```

#### 最短路径算法

对于加权图，找到总权重 (weight)最小的路径是常见要求。

- **迪克斯特拉算法：** 查找图中从单个源节点到所有其他节点的具有非负边权的最短路径。它使用优先队列（最小堆）来高效选择下一个要访问的节点。

  ```python
  import heapq

  def dijkstra(graph, start):
      """使用迪克斯特拉算法计算从起始节点到最短路径。"""
      distances = {node: float('inf') for node in graph}
      distances[start] = 0
      priority_queue = [(0, start)] # (距离, 节点)
      
      while priority_queue:
          current_distance, current_node = heapq.heappop(priority_queue)

          # 节点已经以更短的路径处理过
          if current_distance > distances[current_node]:
              continue

          for neighbor, weight in graph.get(current_node, set()):
              distance = current_distance + weight
              
              # 如果找到到邻居的更短路径
              if distance < distances[neighbor]:
                  distances[neighbor] = distance
                  heapq.heappush(priority_queue, (distance, neighbor))
                  
      return distances

  # 使用加权有向图示例
  shortest_distances = dijkstra(graph_weighted_adj, 'A')
  print(f"\n从A出发的最短距离：{shortest_distances}")
  # 输出：从A出发的最短距离：{'A': 0, 'B': 5, 'C': 3, 'D': 7, 'E': 9, 'F': 4}
  ```

  `NetworkX`提供了 `nx.shortest_path(G, source, target, weight='weight')` 及其他实现迪克斯特拉算法和其他算法的相关函数。

#### 最小生成树 (MST)

MST连接无向加权图中的所有顶点，使其总边权最小，且不形成任何环。普里姆算法和克鲁斯卡尔算法等可找到MST。它们在网络设计、聚类（例如单链接聚类）和图像分割中具有应用。`NetworkX`提供了 `nx.minimum_spanning_tree(G, weight='weight')`。

### 机器学习 (machine learning)中的应用

图结构和算法在多个机器学习方面有直接用途：

1. **推荐系统：** 将用户和物品建模为节点，边代表交互（评分、购买）。图算法可以识别相似用户/物品或预测未来交互。随机游走（例如Node2Vec）等技术从图结构中学习节点嵌入 (embedding)。
2. **社交网络分析：** 识别有影响力的用户（中心性度量），检测社区（如Louvain社区检测算法），并模拟信息传播。
3. **自然语言处理：** 知识图表示从文本中提取的实体和关系。依存分析树是表示语法结构的图。图算法有助于查询知识和理解句子结构。
4. **生物信息学：** 蛋白质-蛋白质相互作用网络、基因调控网络和代谢通路都建模为图。算法有助于识别重要蛋白质/基因或功能模块。
5. **图神经网络 (neural network) (GNNs)：** 一类设计用于直接处理图数据的深度学习 (deep learning)模型。它们通过聚合邻居信息来学习节点表示，从而支持节点分类、链接预测和图分类等任务。PyTorch Geometric (PyG) 和 Deep Graph Library (DGL) 等库促进了GNN的开发。

### 实现考量

"\* **可伸缩性：** 图（社交网络、Web图）可以包含数十亿个节点和边。邻接矩阵变得不可行。稀疏邻接表是首选。对于超大型图，可能需要专门的图数据库（例如Neo4j）或分布式处理框架（例如Apache Spark的GraphX/GraphFrames）。"

- **库选择：** `NetworkX`非常适合通用图分析和中等规模的图。对于性能敏感的任务或大型图，特别是涉及数值计算或GNNs的图，`igraph`、`graph-tool`、`PyG`或`DGL`等库可能提供更好的性能，它们通常利用C/C++后端。

通过将图表示和算法纳入你的工具包，你可以有效地建模和分析关系数据，为解决超越表格或序列数据格式的复杂机器学习 (machine learning)问题提供了高级方法。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: The MIT Press
  一本全面且权威的教材，涵盖了基本数据结构、图表示以及BFS、DFS、Dijkstra和最小生成树等算法。
- [NetworkX Documentation](https://networkx.org/documentation/stable/) — Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart (2025)
  NetworkX的官方文档，这是一个用于创建、操作和研究复杂网络结构与功能的Python库。
- [Graph Neural Networks: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/9079777) — Jie Zhou, Ganqu Cui, Zhengyu Chen, Ming Ding, Stefan Neumann, Jiaying Li, and Jie Tang (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2974937](https://doi.org/10.1109/TNNLS.2020.2974937)
  一篇被广泛引用的综述，全面概述了图神经网络及其方法论和在机器学习中的应用。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons
  一本标准教材，以Python为例并提供实现，介绍了数据结构和算法，包括对图的全面讲解。
