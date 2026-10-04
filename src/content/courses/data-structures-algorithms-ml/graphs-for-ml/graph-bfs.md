---
course: "data-structures-algorithms-ml"
chapter: "graphs-for-ml"
lesson: "graph-bfs"
sourceId: 5300
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-4-graphs-for-ml/graph-bfs"
title: "图遍历：广度优先搜索（BFS）"
description: "了解BFS算法、其特性以及在查找无权重图中最短路径方面的应用。"
order: 2
plots: []
sourceHash: "29abb432245aecb9c352e5c89116268cf56e195c3c9a76183f3f32c7e53bbef1"
sourceCorrections: []
---

图是一种表示关系数据的强大方式。系统地探索这些图表示是各种应用中的常见任务。例如，发现从起始点可达的所有实体、找到两个实体之间的最短连接或理解图中组件的结构，都是重要的目标。图遍历算法提供了有序访问节点的机制来实现这些目标。广度优先搜索（BFS）是一种基本的遍历算法。

想象一下将一颗小石子投入平静的池塘。涟漪会均匀地呈同心圆状扩散。BFS以类似的方式访问图：它从一个选定的源节点开始，首先访问其邻近节点，然后再移动到下一层邻近节点。它逐层访问图。

### BFS算法

BFS通过维护一个待访问节点队列和一个已访问节点集合来工作，以避免循环和重复工作。

其工作方式通常如下：

1. **初始化：**
   - 选择一个起始节点 `s` 来开始遍历。
   - 创建一个队列（通常是先进先出队列），并将起始节点 `s` 入队。
   - 创建一个集合或布尔数组来记录已访问的节点。将 `s` 标记 (token)为已访问。
2. **遍历循环：**
   - 当队列不为空时：
     - 从队列前端取出一个节点 `u`。
     - 处理 `u`（例如，打印其值，检查它是否是目标节点）。
     - 对于 `u` 的每个邻近节点 `v`：
       - 如果 `v` 尚未被访问：
         - 将 `v` 标记为已访问。
         - 将 `v` 入队。

这个过程确保节点按照它们与源节点距离（边数）的递增顺序被访问。这个特性使得BFS适用于查找无权重 (weight)图中的最短路径。

### BFS可视化

我们来追踪在简单图上进行BFS的过程，从节点 'A' 开始。我们将使用邻接表表示：A: [B, D], B: [A, C], C: [B, E], D: [A, E], E: [C, D]。

> 一个简单的无向图。节点 'F' 被包含进来是为了说明，如果从 'A' 开始，它将不会被访问。

**步骤（从A开始）：**

1. **初始化：** `queue = [A]`, `visited = {A}`。
2. **A出队：** 处理A。邻近节点是B、D。
   - B未被访问。标记 (token)B为已访问，B入队。 `queue = [B]`, `visited = {A, B}`。
   - D未被访问。标记D为已访问，D入队。 `queue = [B, D]`, `visited = {A, B, D}`。
3. **B出队：** 处理B。邻近节点是A、C。
   - A已访问。跳过。
   - C未被访问。标记C为已访问，C入队。 `queue = [D, C]`, `visited = {A, B, D, C}`。
4. **D出队：** 处理D。邻近节点是A、E。
   - A已访问。跳过。
   - E未被访问。标记E为已访问，E入队。 `queue = [C, E]`, `visited = {A, B, D, C, E}`。
5. **C出队：** 处理C。邻近节点是B、E。
   - B已访问。跳过。
   - E已访问。跳过。 `queue = [E]`, `visited = {A, B, D, C, E}`。
6. **E出队：** 处理E。邻近节点是C、D。
   - C已访问。跳过。
   - D已访问。跳过。 `queue = []`, `visited = {A, B, D, C, E}`。
7. **队列为空：** 停止。

节点访问（处理）的顺序是A、B、D、C、E。请注意，所有与A距离为1的节点（B、D）都在与A距离为2的节点（C、E）之前被访问。节点F从未被访问，因为它无法从A到达。

### 伪代码

以下是更正式的伪代码表示：

```
BFS(graph, start_node):
  设 Q 为一个队列
  设 visited 为一个集合

  将 start_node 加入 Q
  将 start_node 加入 visited

  当 Q 不为空时:
    current_node = Q.出队()

    // 处理 current_node（例如，打印、检查条件）
    process(current_node)

    对于 graph.get_neighbors(current_node) 中的每个 neighbor:
      如果 neighbor 不在 visited 中:
        将 neighbor 加入 visited
        将 neighbor 加入 Q
```

### Python实现

使用Python的 `collections.deque` 进行高效队列实现和字典作为邻接表是常见的做法：

```python
from collections import deque

def bfs(graph, start_node):
    """
    在图上执行广度优先搜索。

    参数:
        graph (dict): 图的邻接表表示
                      （例如，{'A': ['B', 'D'], ...}）。
        start_node: 开始遍历的节点。

    返回:
        list: 节点被访问的顺序列表。
              如果 start_node 不在图中，返回空列表。
    """
    if start_node not in graph:
        return [] # 起始节点必须在图中

    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    visited_order = []

    while queue:
        current_node = queue.popleft() # 从左侧出队
        visited_order.append(current_node)

        # 如果需要，在这里处理节点（例如，print(current_node)）

        # 将邻近节点加入队列
        # 使用 graph.get(current_node, []) 处理没有出边的节点
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor) # 从右侧入队

    return visited_order

# 使用可视化中的图进行示例：
graph_example = {
    'A': ['B', 'D'],
    'B': ['A', 'C'],
    'C': ['B', 'E'],
    'D': ['A', 'E'],
    'E': ['C', 'D'],
    'F': [] # 节点 F 未连接到主组件
}

start = 'A'
traversal_path = bfs(graph_example, start)
print(f"BFS traversal starting from {start}: {traversal_path}")
# 预期输出: 从A开始的BFS遍历: ['A', 'B', 'D', 'C', 'E']

start_unconnected = 'F'
traversal_unconnected = bfs(graph_example, start_unconnected)
print(f"BFS traversal starting from {start_unconnected}: {traversal_unconnected}")
# 预期输出: 从F开始的BFS遍历: ['F']
```

### 性能分析

了解BFS的性能对于选择合适的算法很重要。

- **时间复杂度：** 在最坏情况下（假设使用邻接表表示），BFS 会且仅会访问每个节点和每条边一次。

  - 每个节点被入队和出队一次：$O(V)$ 操作，其中 $V$ 是顶点（节点）的数量。
  - 处理节点 `u` 时，我们遍历其所有邻近节点。在整个执行过程中，邻近节点检查的总数对应于所有已访问节点的度数之和。对于有向图，这是已访问节点的出边数量。对于无向图，每条边 $(u, v)$ 会被考虑两次（一次从 $u$ 考虑，一次从 $v$ 考虑）。无论哪种情况，与边相关的总工作量与正在遍历的连通分量中的边数 $E$ 成比例。
  - 因此，总的时间复杂度是 $O(V + E)$。
- **空间复杂度：** 所需空间主要由 `visited` 集合和 `queue` 的存储占用。

  - `visited` 集合最多可以存储 $V$ 个节点。
  - 在最坏情况下，`queue` 可能需要容纳特定层次的所有节点。对于非常密集的图或星形图，这可能接近 $O(V)$ 个节点。
  - 因此，空间复杂度通常是 $O(V)$。

### 机器学习 (machine learning)应用

虽然BFS不总是直接作为模型核心训练循环（如梯度下降 (gradient descent)）的一部分，但它在机器学习从业者的工具箱中是处理图相关任务的有用的工具：

1. **无权重 (weight)图中的最短路径：** 这是经典应用。如果边代表相同“成本”的连接（例如，直接的友谊链接，单次交互步骤），BFS可以找到两个节点之间边数最少的路径。这在推荐系统（用户与物品之间通过交互的最短路径）或社交网络分析（六度分隔）中很有用。
2. **网络查看与分析：** BFS用于找到从起始点可达的所有节点，识别大型图中的连通分量，或检查节点之间的连通性。这是理解图数据集结构的基础。
3. **网络爬虫（类比）：** 搜索引擎爬虫通常使用类似BFS的策略来寻找网页，在深入之前首先访问从当前页面直接链接的页面。
4. **特征工程：** 通过BFS计算的特定节点（或节点集合）的距离有时可以用作机器学习模型的特征。例如，在欺诈检测中，与已知欺诈用户的最短距离可能是一个有帮助的特征。
5. **种子集扩展：** 在基于图的半监督学习 (supervised learning) (semi-supervised learning)或聚类中，BFS可以从初始的标记 (token)节点（种子）扩展，找到可能属于同一类别或聚类的邻近节点。

BFS提供了系统地、逐层地查看图结构的方法。它在无权重图中找到最短路径的能力以及其可预测的 $O(V+E)$ 性能使其成为分析关系数据图的基础算法。接下来我们将了解深度优先搜索（DFS），它使用不同的策略来遍历图。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 22, Section 22.2 (pp. 609-618)
  一本广受赞誉且全面的算法教材，对 BFS 及其特性和复杂性分析进行了严谨的数学处理。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons; Pages: Chapter 14, Section 14.3.1 (pp. 641-645)
  一本广泛使用的教材，提供了 BFS 的清晰解释和 Python 实现，适合寻找实用编码示例的读者。
- [Introduction to Algorithms, Lecture 14: Breadth-First Search (BFS)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/) — Erik Demaine, Srini Devadas (2011)
  Journal: MIT OpenCourseWare; Publisher: MIT OpenCourseWare
  麻省理工学院开放课程讲义，对 BFS 算法、特性和应用提供了易懂而严谨的学术解释。
