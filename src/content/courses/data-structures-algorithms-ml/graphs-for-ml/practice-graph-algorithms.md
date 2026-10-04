---
course: "data-structures-algorithms-ml"
chapter: "graphs-for-ml"
lesson: "practice-graph-algorithms"
sourceId: 5314
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-4-graphs-for-ml/practice-graph-algorithms"
title: "动手实践：图表示与遍历"
description: "实现图数据结构，并在示例图上执行 BFS 和 DFS 遍历。"
order: 7
plots: []
sourceHash: "32df0adba8e5ee4134eba235faae77b7a9192db1c2f7e11536173d94884db7a4"
sourceCorrections: []
---

图表示和遍历通过实践得以实现。使用一个小型示例图，在 Python 中构建其结构并应用遍历算法。这种动手实践有助于理解不同表示方法之间的权衡以及 BFS 和 DFS 的运作方式。

### 定义示例图

设想一个小型社交网络，用户之间如果互为好友则相互连接。我们将把这个网络表示为一个无向图。

节点：Alice, Bob, Charlie, David, Eve
边：

- Alice ↔ Bob
- Alice ↔ Charlie
- Bob ↔ David
- Charlie ↔ David
- Charlie ↔ Eve
- David ↔ Eve

以下是我们简单社交网络图的示意图：

> 一个表示五位用户之间友谊的简单无向图。

### 实现图表示

我们将使用邻接表和邻接矩阵两种方式来实现这个图。

#### 邻接表表示

邻接表通常更受稀疏图（边相对可能总数较少的图）的青睐，因为它节省空间。我们可以用 Python 字典来表示它，键是节点，值是其邻居列表。

```python
# 邻接表表示
graph_adj_list = {
    'Alice': ['Bob', 'Charlie'],
    'Bob': ['Alice', 'David'],
    'Charlie': ['Alice', 'David', 'Eve'],
    'David': ['Bob', 'Charlie', 'Eve'],
    'Eve': ['Charlie', 'David']
}

# 使用示例：获取 Charlie 的邻居
print(f"Charlie 的邻居：{graph_adj_list['Charlie']}")

# 使用示例：检查 Alice 和 David 是否连接
is_connected = 'David' in graph_adj_list['Alice'] # 需要检查邻居列表
print(f"Alice 和 David 是否直接连接？{is_connected}")
```

这种表示方法使得查找给定节点的所有邻居变得容易（查找时间与邻居数量成正比）。然而，检查两个特定节点之间是否存在边可能需要搜索其中一个节点的邻居列表。添加或删除边通常效率较高。

#### 邻接矩阵表示

邻接矩阵使用一个二维数组，其中行 `i`、列 `j` 的值表示节点 `i` 和节点 `j` 之间是否存在边（为 1 或边的权重 (weight)），否则为 0。这要求将节点名称映射到整数索引。

```python
import numpy as np

nodes = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
node_to_index = {node: i for i, node in enumerate(nodes)}
num_nodes = len(nodes)

# 初始化一个空矩阵
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# 根据边填充矩阵（无向图为对称矩阵）
edges = [
    ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Bob', 'David'),
    ('Charlie', 'David'), ('Charlie', 'Eve'), ('David', 'Eve')
]

for u, v in edges:
    u_idx, v_idx = node_to_index[u], node_to_index[v]
    adj_matrix[u_idx, v_idx] = 1
    adj_matrix[v_idx, u_idx] = 1 # 无向图

print("邻接矩阵：")
print(adj_matrix)
print(f"\n节点映射：{node_to_index}")

# 使用示例：检查 Alice 和 David 是否连接
alice_idx = node_to_index['Alice']
david_idx = node_to_index['David']
is_connected_matrix = adj_matrix[alice_idx, david_idx] == 1
print(f"\nAlice 和 David 是否直接连接（矩阵）？{is_connected_matrix}")

# 使用示例：检查 Alice 和 Bob 是否连接
bob_idx = node_to_index['Bob']
is_connected_matrix_ab = adj_matrix[alice_idx, bob_idx] == 1
print(f"Alice 和 Bob 是否直接连接（矩阵）？{is_connected_matrix_ab}")
```

邻接矩阵提供了非常快的 O(1) 边检查速度，可以检查任意两个节点之间是否存在边。然而，它需要 $O(V^2)$ 的空间，其中 $V$ 是顶点数量，这对于稀疏图来说效率可能不高。遍历一个节点的所有邻居需要检查整行，耗时 $O(V)$。添加或删除节点代价较高，因为它需要重新调整矩阵大小。

### 实现图遍历

现在，让我们使用邻接表表示来实现 BFS 和 DFS。

#### 广度优先搜索 (BFS)

BFS 逐层搜索图。它使用队列来记录待访问的节点。

```python
from collections import deque

def bfs(graph, start_node):
    """对用邻接表表示的图执行 BFS。"""
    if start_node not in graph:
        print(f"起始节点 '{start_node}' 未在图中找到。")
        return []

    visited = set()         # 记录已访问的节点
    queue = deque([start_node]) # 使用起始节点初始化队列
    visited.add(start_node)
    traversal_order = []    # 存储访问节点的顺序

    while queue:
        current_node = queue.popleft() # 出队一个节点
        traversal_order.append(current_node)
        # print(f"Visiting: {current_node}") # 可选：打印访问步骤

        # 将未访问的邻居入队
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return traversal_order

# 从 'Alice' 开始执行 BFS
bfs_path = bfs(graph_adj_list, 'Alice')
print(f"\n从 Alice 开始的 BFS 遍历：{bfs_path}")

# 从 'David' 开始执行 BFS
bfs_path_david = bfs(graph_adj_list, 'David')
print(f"从 David 开始的 BFS 遍历：{bfs_path_david}")
```

BFS 对于在无权图中根据边数查找最短路径很有用。请注意它是如何先检查离起始节点较近的节点（'Bob' 和 'Charlie'），然后再移动到更远的节点（'David'、'Eve'）。

#### 深度优先搜索 (DFS)

DFS 沿着每条分支尽可能行进，然后回溯。它可以递归实现，也可以使用栈迭代实现。以下是一个递归实现：

```python
def dfs_recursive(graph, start_node, visited=None, traversal_order=None):
    """对用邻接表表示的图递归执行 DFS。"""
    if visited is None:
        visited = set()
    if traversal_order is None:
        traversal_order = []

    if start_node not in graph:
        print(f"起始节点 '{start_node}' 未在图中找到。")
        return []

    visited.add(start_node)
    traversal_order.append(start_node)
    # print(f"Visiting: {start_node}") # 可选：打印访问步骤

    for neighbor in graph.get(start_node, []):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, traversal_order)

    return traversal_order

# 从 'Alice' 开始执行 DFS
visited_set_dfs = set() # 如果需要后续调用，则需在外部管理已访问集合
dfs_path = dfs_recursive(graph_adj_list, 'Alice', visited_set_dfs)
print(f"\n从 Alice 开始的 DFS 遍历：{dfs_path}")

# 从 'David' 开始执行 DFS
visited_set_dfs_david = set()
dfs_path_david = dfs_recursive(graph_adj_list, 'David', visited_set_dfs_david)
print(f"从 David 开始的 DFS 遍历：{dfs_path_david}")
```

DFS 路径可能因邻居处理顺序而异。在我们从 'Alice' 开始的例子中，它可能会通过 'Bob' 行进到 'David'，然后再处理 'Charlie' 及其通向 'Eve' 的分支。DFS 适用于循环检测、拓扑排序（针对有向无环图）以及检查连通性等任务。

### 实用考量

- **表示方法选择：** 对于社交网络或推荐系统中常见的稀疏图，邻接表通常更节省内存，并且在涉及邻居的操作中通常更快。邻接矩阵擅长快速查找边，但占用更多空间。
- **遍历应用场景：** BFS 是查找最短路径（在无权图中）或进行分层处理的理想选择。DFS 适用于路径查找、循环检测以及需要完全检查分支的问题。
- **库：** 对于 Python 中更大或更复杂的图任务，`NetworkX` 等库提供了预构建的图结构和算法，节省了实现时间并提供了优化的性能。然而，理解其底层实现（如本次实践所示）有助于有效选择和使用这些工具。

本次实践练习展示了如何表示图并实现基本的遍历算法。这些方法是许多基于图的机器学习 (machine learning)应用的根本，例如分析网络结构、构建推荐引擎或执行社区检测。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  一本标准教科书，介绍了图的表示方法（邻接表、邻接矩阵）和遍历算法（BFS、DFS），提供了理论基础。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons
  提供了图数据结构和算法的Python实现，补充了本节的代码示例和实践方法。
- [NetworkX Documentation](https://networkx.org/documentation/stable/) — Aric A. Hagberg, Daniel A. Schult, Pieter J. Swart (2025)
  NetworkX的官方文档，这是一个用于图创建、操作和算法应用的常用Python库。
- [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) — Erik Demaine, Jason Ku, Justin Solomon (2020)
  Publisher: MIT OpenCourseWare
  大学课程材料，涵盖了基础算法设计与分析，包括图、它们的表示方法、BFS和DFS。
