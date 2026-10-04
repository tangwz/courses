# 图遍历：深度优先搜索（DFS）

来源：[原文](https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-4-graphs-for-ml/graph-dfs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

广度优先搜索（BFS）逐层遍历图，而深度优先搜索（DFS）采用不同的方式。顾名思义，DFS 在回溯之前会沿着每条路径尽可能深入。可以想象成在迷宫中穿行：沿着一条路径走到死胡同，然后回溯到上一个交叉点，尝试另一条未被检查的路径。这种策略对多种图算法来说非常有用，可用于分析网络结构和依赖关系。

### DFS 算法

DFS 通常从任意节点（在搜索中常被称为“源”或“根”）开始，并沿某条路径进行。以下是一般过程：

1. **开始：** 选择一个起始节点。将其标记 (token)为已访问。
2. **遍历：** 移动到一个相邻且未访问的节点。将其标记为已访问。
3. **向下继续：** 重复步骤2。继续移动到*当前*节点的未访问邻居。
4. **回溯：** 如果当前节点没有未访问的邻居，回溯到您到达当前节点的那个节点。
5. **继续：** 从您回溯到的节点，检查它是否有其他未访问的邻居。如果有，从步骤2开始重复，使用其中一个邻居。如果没有，则进一步回溯。
6. **重复：** 继续此过程，直到所有从起始节点可达的节点都被访问过。
7. **不连通图：** 如果图可能不连通（这意味着并非所有节点都可从单个起始节点到达），您可能需要从任何剩余的未访问节点开始重复DFS过程，直到图中所有节点都被访问。

与 BFS 使用队列管理要访问的节点不同，DFS 自然地使用栈。这个栈可以是程序的调用栈（在递归实现中），也可以是显式栈数据结构（在迭代实现中）。

### 递归实现

递归提供了一种巧妙的方法来实现 DFS。函数会为其每个未访问的邻居调用自身，有效地利用调用栈管理回溯过程。

假设我们的图使用邻接表（一个字典，键是节点，值是其邻居列表）表示。

```python
import collections

def dfs_recursive(graph, node, visited):
    """
    从'node'开始递归执行深度优先搜索。

    参数：
        graph (dict): 图的邻接表表示。
                      示例：{'A': ['B', 'C'], 'B': ['D'], ...}
        node: 当前DFS操作的起始节点。
        visited (set): 用于记录已访问节点的集合。

    返回：
        无。就地修改'visited'集合。
    """
    if node not in visited:
        print(f"正在访问节点：{node}") # 处理节点（例如，打印）
        visited.add(node)
        if node in graph: # 检查节点是否有邻居
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs_recursive(graph, neighbor, visited)

# 示例用法：
graph_adj = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("从节点A开始的递归DFS：")
visited_nodes_rec = set()
dfs_recursive(graph_adj, 'A', visited_nodes_rec)
print("已访问节点：", visited_nodes_rec)

# 处理不连通图（如果需要）
# 您可以遍历所有节点，如果未访问则启动DFS
# all_nodes = list(graph_adj.keys()) # 或通过其他方式获取所有节点
# visited_all = set()
# for node in all_nodes:
#    if node not in visited_all:
#        print(f"\n从{node}开始新的DFS组件")
#        dfs_recursive(graph_adj, node, visited_all)
```

递归方式与 DFS 的定义相符：访问一个节点，然后逐个递归访问其未访问的邻居。

### 使用栈的迭代实现

虽然递归对 DFS 来说通常很直观，但对于非常深的图，它可能导致栈溢出错误。使用显式栈的迭代方法可避免此限制。

```python
import collections

def dfs_iterative(graph, start_node):
    """
    使用栈迭代执行深度优先搜索。

    参数：
        graph (dict): 图的邻接表表示。
        start_node: 搜索的起始节点。

    返回：
        set: 一个集合，包含搜索期间访问过的所有节点。
    """
    if start_node not in graph:
         print(f"起始节点 {start_node} 不在图中。")
         return set()

    visited = set()
    stack = collections.deque([start_node]) # 将 deque 用作栈

    print("从节点A开始的迭代DFS：")
    while stack:
        node = stack.pop() # 获取最后添加的节点（后进先出）

        if node not in visited:
            print(f"正在访问节点：{node}") # 处理节点
            visited.add(node)

            # 将未访问的邻居添加到栈中
            # 以反向顺序添加邻居，以便按标准顺序访问它们
            # （尽管顺序对正确性并非严格必要）
            if node in graph:
                 # 如果需要模拟递归顺序，则反向处理邻居
                 for neighbor in reversed(graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)

    return visited

# 示例用法（使用与之前相同的graph_adj）：
visited_nodes_iter = dfs_iterative(graph_adj, 'A')
print("已访问节点：", visited_nodes_iter)
```

在迭代版本中，我们将起始节点压入栈。然后，只要栈不为空，我们就弹出一个节点，访问它（如果尚未访问），并将其未访问的邻居压入栈。在 Python 中，使用 `collections.deque` 进行高效的栈操作（从同一端添加和弹出）很常见。请注意，邻居被压入栈的顺序会影响 DFS 遍历的具体路径，但从起始点可达的已访问节点集合最终将是相同的。

### DFS 遍历可视化

考虑下面这个简单图。让我们从节点 'A' 开始追踪 DFS 路径，假设邻居按字母顺序进行检查。

> 从节点A开始的一种可能的DFS遍历（按字母顺序检查邻居）：A -> B -> D -> E -> F -> C -> G。红色节点是起始点。带数字的粉色边显示了访问未访问节点的顺序。

路径深入：从 A 到 B，从 B 到 D。D 没有未访问的邻居，因此回溯到 B。B 的下一个未访问邻居是 E。前进 A -> B -> E。E 的下一个未访问邻居是 F。前进 A -> B -> E -> F。F 的邻居（C、E、A）都已被访问或在当前回到起点的路径上，因此回溯到 E，然后 B，再然后 A。A 的下一个未访问邻居是 C。前进 A -> C。C 的下一个未访问邻居是 F（已访问）。C 的下一个未访问邻居是 G。前进 A -> C -> G。G 没有未访问的邻居。回溯到 C，然后 A。A 的所有邻居都已访问。DFS 完成。访问顺序将是 A、B、D、E、F、C、G。

### 复杂度分析

- **时间复杂度：** DFS 对每个节点和每条边最多访问一次（在有向图中）或两次（在无向图中，每个方向一次）。因此，时间复杂度是 $O(V + E)$，其中 $V$ 是顶点（节点）的数量，$E$ 是边的数量。这是因为我们标记 (token)节点为已访问，不再重复处理它们，并且我们检查从已访问节点发出的每条边。
- **空间复杂度：** 空间复杂度取决于实现方式。
  - 对于递归实现，空间复杂度由递归栈的最大深度决定，在最坏情况（一个长链状图）下可以是 $O(V)$。
  - 对于使用显式栈的迭代实现，栈的最大大小在最坏情况下也可以是 $O(V)$。
  - 我们还需要空间来存储 `visited` 集合，这需要 $O(V)$ 空间。
    因此，总空间复杂度通常由栈或递归深度决定，最终为 $O(V)$。

### DFS 的应用

DFS 这种偏向于深度行进的特性使其适用于多种图相关任务，这些任务常见于机器学习 (machine learning)背景中：

1. **环检测：** DFS 可以轻松检测图中的环。如果在遍历期间，我们遇到一个已经访问过的节点，并且该节点当前位于递归栈中（或在迭代方法中正在被主动处理），则存在一个环。这对分析依赖关系、确保某些模型中的有向无环图（DAG）或识别反馈循环很重要。
2. **拓扑排序：** 对于有向无环图（DAG），DFS 可以生成节点的线性顺序，使得对于从节点 $u$ 到节点 $v$ 的每条有向边，$u$ 在顺序中位于 $v$ 之前。这在调度有依赖关系的任务、解决特征工程管道中的依赖问题或理解贝叶斯网络中节点的重要性方面是基础。拓扑排序可以通过记录节点完成 DFS 遍历的顺序（即，在其所有邻居都被访问之后）来获得。
3. **寻找连通分量：** 在无向图中，您可以从任意一个未访问的节点开始运行 DFS，以找出属于其连通分量的所有节点。通过对任何剩余的未访问节点重复此过程，可以识别图中的所有连通分量。这对于将数据表示为图时的聚类或分区很有用。
4. **路径查找：** DFS 如果存在，可以找到两节点之间的一条路径。它不保证找到最短路径（那是 BFS 在无权图中的优势），但在确定可达性方面很有效。

### DFS 与 BFS 总结

- **遍历策略：** DFS 深入；BFS 逐层移动。
- **数据结构：** DFS 使用栈（隐式通过递归或显式）；BFS 使用队列。
- **最短路径（无权图）：** BFS 找到最短路径。DFS 不保证这一点。
- **内存使用：** 如果图宽但不深（栈中节点较少），DFS 可能比 BFS 更节省内存。如果图深但不宽（任何层级队列中节点较少），BFS 可能更好。在最坏情况下，两者都可能需要 $O(V)$ 空间。
- **常见用途：** DFS 常用于环检测、拓扑排序和遍历迷宫状结构。BFS 更适用于最短路径和寻找最接近源的节点。

理解 DFS 提供了另一个重要的工具，用于遍历和分析支撑许多机器学习 (machine learning)问题的图结构，从特征依赖到网络分析和推荐系统。

## 参考资料

- [Introduction to Algorithms, 4th Edition](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 22, Section 22.3, "Depth-First Search"
  一本关于算法设计和分析的基础教材，全面涵盖了图遍历算法。
- [Algorithms, 4th Edition](https://algs4.cs.princeton.edu/) — Robert Sedgewick and Kevin Wayne (2011)
  Publisher: Pearson; Pages: Chapter 4, Section 4.1, "Undirected Graphs" (includes DFS)
  关于基础算法的全面资源，包含实际应用和实现。
- [6.006 Introduction to Algorithms, Fall 2011 - Lecture 12: Depth-First Search (DFS)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-14-depth-first-search-dfs-topological-sort/) — Erik Demaine, Srini Devadas (2011)
  Publisher: MIT OpenCourseWare
  提供作为算法入门课程一部分的深度优先搜索讲义和视频。

---

[上一节](02-%E5%9B%BE%E9%81%8D%E5%8E%86%EF%BC%9A%E5%B9%BF%E5%BA%A6%E4%BC%98%E5%85%88%E6%90%9C%E7%B4%A2%EF%BC%88BFS%EF%BC%89.md) · [下一节](04-%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84%E7%AE%97%E6%B3%95%E6%A6%82%E8%BF%B0.md)
