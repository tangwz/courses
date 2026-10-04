---
course: "advanced-python-programming-ml"
chapter: "advanced-data-structures-algorithms-ml"
lesson: "practice-implementing-kd-tree"
sourceId: 2340
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/practice-implementing-kd-tree"
title: "实践操作：实现用于最近邻搜索的 k-d 树"
description: "从头开始用 Python 实现 k-d 树数据结构，以加速最近邻搜索。"
order: 7
plots: ["plots/2340-0.json"]
sourceHash: "aa071a2fe9c5fc12ae3a8ab35aaf040924313d2de36388599bd07de9bd1b82cb"
sourceCorrections: []
---

从头开始用 Python 实现一个 k-d 树（k维树）提供了基本数据结构如何支持高效的机器学习 (machine learning)算法的实际应用。k-d 树对于加速最近邻搜索特别有用，这是 k-最近邻 (k-NN) 算法、密度估计以及某些聚类方法中常见的操作。虽然像 Scikit-learn 这样的库提供了优化过的实现，但自己构建一个能对它的内部机制和优缺点提供有价值的理解。

暴力最近邻搜索会将查询点与数据集中的每个其他点进行比较，对于 $k$ 维空间中的 $n$ 个点，时间复杂度为 $O(n \cdot k)$。对于很大的数据集，这会变得计算上难以承受。k-d 树的目标是减少这种搜索时间，通过递归划分数据空间，在低维空间中通常达到平均 $O(\log n)$ 的复杂度。

### k-d 树的结构说明

k-d 树是一种二叉树，其中每个节点表示一个轴对齐 (alignment)的超平面，将空间分成两个半空间。点可以隐式地存储在这些区域内，或显式地存储在节点上。

1. **划分维度**：在树的每个层级，我们选择一个维度进行划分。一种常用方法是循环选择维度（0, 1, ..., k-1, 0, 1, ...）。
2. **划分值**：对于选定的维度，我们选择一个枢轴点，通常是沿该维度的中位数点。这个点存储在当前节点中。
3. **分区**：所有在划分维度上值小于枢轴点值的点进入左子树，所有值大于或等于的点进入右子树。
4. **递归**：这个过程对左子集和右子集递归重复，直到节点只包含一个点或为空（叶节点）。

让我们定义一个简单结构来表示树中的节点。`namedtuple` 对此很方便：

```python
import collections
import numpy as np
from typing import List, Tuple, Optional, Any

Point = np.ndarray # 假设点是 NumPy 数组

class Node(collections.namedtuple('Node', ['point', 'split_dim', 'left', 'right'])):
    """
    表示 k-d 树中的一个节点。

    属性:
        point (Optional[Point]): 存储在此节点的数据点（如果非空）。
                                  对于内部节点，这是划分点。
        split_dim (Optional[int]): 此节点用于划分的维度。
        left (Optional[Node]): 左子节点。
        right (Optional[Node]): 右子节点。
    """
    pass

# 为了清晰起见，将属性设为可选，尽管 split_dim 和 point
# 通常会为内部节点设置。NamedTuple 的默认值不易
# 修改，因此我们在创建逻辑中处理 Optional。
Node.__new__.__defaults__ = (None,) * len(Node._fields)
```

### 构建树

k-d 树的核心是递归构建函数。它接收一个点列表和当前深度（用于确定划分维度）。

```python
def build_kdtree(points: List[Point], depth: int = 0) -> Optional[Node]:
    """
    从点列表递归构建 k-d 树。

    参数:
        points: k 维点的列表（NumPy 数组）。
        depth: 树中的当前深度（用于循环维度）。

    返回:
        构建的 k-d 树的根节点，如果 points 为空则返回 None。
    """
    if not points:
        return None

    k = len(points[0]) # 数据的维度
    split_dim = depth % k

    # 沿划分维度对点进行排序并选择中位数
    points.sort(key=lambda p: p[split_dim])
    median_idx = len(points) // 2
    median_point = points[median_idx]

    # 递归构建左右子树
    left_subtree = build_kdtree(points[:median_idx], depth + 1)
    right_subtree = build_kdtree(points[median_idx + 1:], depth + 1)

    return Node(
        point=median_point,
        split_dim=split_dim,
        left=left_subtree,
        right=right_subtree
    )

# 示例用法（假设是二维点）：
# points_list = [np.array([2, 3]), np.array([5, 4]), np.array([9, 6]),
#                np.array([4, 7]), np.array([8, 1]), np.array([7, 2])]
# root_node = build_kdtree(points_list.copy()) # 如果需要原始列表，请传入副本
```

此实现使用中位数点来划分数据，旨在构建平衡树。每一步的排序都会带来每层 $O(n \log n)$ 的复杂度。由于有 $\log n$ 层，总构建时间是 $O(k \cdot n \log^2 n)$。使用像 `introselect` （`numpy.partition` 内部使用的算法）这样的中位数查找算法而不是完全排序，可以将其改进到 $O(k \cdot n \log n)$，但为了简单起见，这里我们使用排序。

### 搜索最近邻

查找最近邻涉及高效遍历树，剪除不可能包含比目前找到的最佳点更近的点的分支。

1. **下降**：根据查询点相对于每个节点划分维度的坐标，沿着树向下遍历，直到到达叶节点。这个叶节点成为最初的 `best_guess`。
2. **回溯和剪枝**：从叶节点开始，递归地回溯树：
   - 在每个节点，比较查询点到该节点点的距离。如果它比当前的 `best_guess` 更近，则更新 `best_guess`。
   - 重要的是，检查以查询点为中心、半径等于当前 `best_distance` 的超球体是否与当前节点定义的划分超平面相交。查询点到超平面的距离就是其在 `split_dim` 坐标上的绝对差值：`abs(query_point[split_dim] - node.point[split_dim])`。
   - 如果此距离小于 `best_distance`，则另一个子树（在下降过程中未初步访问的那个）*可能*包含一个更近的点，因此也递归搜索该子树。否则，剪除该分支。

我们将使用一个辅助函数来管理搜索状态。使用列表或自定义类实例等可变对象来表示 `best_guess` 和 `best_distance`，可以避免 Python 在递归中处理不可变类型时出现的问题。我们将使用一个简单的列表 `[best_point, best_dist_sq]` 来存储找到的最佳点及其平方距离（直到最后才进行平方根运算会更高效）。

```python
import math

def squared_distance(p1: Point, p2: Point) -> float:
    """计算两点之间的欧几里得距离平方。"""
    return np.sum((p1 - p2)**2)

def find_nearest_neighbor_recursive(node: Optional[Node], query_point: Point, depth: int, best: List[Any]):
    """
    用于最近邻搜索的递归辅助函数。

    参数:
        node: 当前正在访问的节点。
        query_point: 要查找最近邻的点。
        depth: 树中的当前深度。
        best: 一个列表，包含 [找到的最佳点, 找到的最佳平方距离]。
              原地更新。
    """
    if node is None:
        return

    k = len(query_point)
    split_dim = depth % k # 或者如果已存储则使用 node.split_dim

    # 计算查询点到当前节点点的平方距离
    dist_sq = squared_distance(query_point, node.point)

    # 如果当前节点更近，则更新 best
    if dist_sq < best[1]:
        best[0] = node.point
        best[1] = dist_sq

    # 确定首先探索哪个子树
    diff = query_point[split_dim] - node.point[split_dim]
    if diff < 0:
        close_branch, away_branch = node.left, node.right
    else:
        close_branch, away_branch = node.right, node.left

    # 首先递归搜索较近的子树
    find_nearest_neighbor_recursive(close_branch, query_point, depth + 1, best)

    # 检查“远离”分支是否需要探索（剪枝步骤）
    # 比较沿划分维度的平方距离与最佳距离
    if (diff**2) < best[1]:
        find_nearest_neighbor_recursive(away_branch, query_point, depth + 1, best)

def find_nearest_neighbor(root: Optional[Node], query_point: Point) -> Tuple[Optional[Point], float]:
    """
    在 k-d 树中查找查询点的最近邻。

    参数:
        root: k-d 树的根节点。
        query_point: 要查找最近邻的点。

    返回:
        一个元组，包含（最近点，距离），如果树为空则为（None，无穷大）。
    """
    if root is None:
        return None, float('inf')

    # 用一个很大的距离初始化 best
    # 为了搜索效率，我们存储平方距离
    best_state = [None, float('inf')]
    find_nearest_neighbor_recursive(root, query_point, 0, best_state)

    # 返回点和实际的欧几里得距离
    return best_state[0], math.sqrt(best_state[1])

# 示例用法：
# query = np.array([6, 5])
# nearest, distance = find_nearest_neighbor(root_node, query)
# print(f"查询点: {query}")
# print(f"最近点: {nearest}, 距离: {distance:.4f}")
```

### 扩展到 k-最近邻 (k-NN)

将搜索扩展到 $k$ 个最近邻需要维护一个包含目前找到的 $k$ 个最佳候选的列表（或更优地，一个有界优先队列/最大堆）。剪枝条件有所变化：只有当查询点到划分超平面的距离大于当前 $k$ 个候选集中*最远*点的距离时，我们才剪除该分支。

我们可以使用 Python 的 `heapq` 模块，它实现了一个最小堆。为了模拟一个大小为 $k$ 的距离最大堆，我们可以在堆中存储*负*平方距离。当堆的大小超过 $k$ 时，我们移除最小的负平方距离（它对应于最大的实际平方距离）。

```python
import heapq

def find_k_nearest_neighbors(root: Optional[Node], query_point: Point, k: int) -> List[Tuple[float, Point]]:
    """
    在 k-d 树中查找查询点的 k 个最近邻。

    参数:
        root: k-d 树的根节点。
        query_point: 要查找最近邻的点。
        k: 要查找的邻居数量。

    返回:
        一个元组列表（距离，点），按距离排序。
        如果树为空或 k <= 0，则返回空列表。
    """
    if root is None or k <= 0:
        return []

    # 使用存储（-平方距离，点）的最小堆来模拟距离的最大堆
    # 这会保留距离*最小*（负距离最大）的 k 个点
    knn_heap: List[Tuple[float, Point]] = []

    _find_knn_recursive(root, query_point, 0, k, knn_heap)

    # 将堆结果转换为按距离排序的（距离，点）列表
    result = [(-neg_dist_sq, point) for neg_dist_sq, point in knn_heap]
    result.sort(key=lambda x: x[0]) # 按实际距离排序
    # 计算实际距离
    final_result = [(math.sqrt(dist_sq), point) for dist_sq, point in result]

    return final_result

def _find_knn_recursive(node: Optional[Node], query_point: Point, depth: int, k: int, knn_heap: List[Tuple[float, Point]]):
    """使用堆进行 k-NN 搜索的递归辅助函数。"""
    if node is None:
        return

    num_dims = len(query_point)
    split_dim = depth % num_dims

    # 计算到当前节点点的平方距离
    dist_sq = squared_distance(query_point, node.point)
    neg_dist_sq = -dist_sq

    # 将当前点添加到堆/更新堆
    if len(knn_heap) < k:
        heapq.heappush(knn_heap, (neg_dist_sq, node.point))
    elif neg_dist_sq > knn_heap[0][0]: # 当前点比堆中最远的点更近
        heapq.heapreplace(knn_heap, (neg_dist_sq, node.point))

    # 确定分支
    diff = query_point[split_dim] - node.point[split_dim]
    if diff < 0:
        close_branch, away_branch = node.left, node.right
    else:
        close_branch, away_branch = node.right, node.left

    # 搜索较近的分支
    _find_knn_recursive(close_branch, query_point, depth + 1, k, knn_heap)

    # 检查“远离”分支是否需要搜索
    # 剪枝条件：比较到超平面的距离与到第 K 个邻居的距离
    # 第 K 个邻居的负平方距离是 knn_heap[0][0]（最小元素）
    farthest_dist_sq = -knn_heap[0][0] if knn_heap else float('inf')

    if (diff**2) < farthest_dist_sq or len(knn_heap) < k:
         _find_knn_recursive(away_branch, query_point, depth + 1, k, knn_heap)

# 示例用法：
# K = 3
# neighbors = find_k_nearest_neighbors(root_node, query, K)
# print(f"\n查找点: {query} 的 {K} 个最近邻")
# for dist, point in neighbors:
#     print(f"  点: {point}, 距离: {dist:.4f}")
```

### 实际考量与可视化

虽然 k-d 树在低维空间（例如，2D、3D）中大幅提高搜索速度，但随着维度数量 ($k$) 的增加，它们的性能会下降。这通常被称为“维度灾难”。在高维空间 (high-dimensional space)中，由划分定义的超立方体在剪枝搜索空间方面的效果会降低，并且搜索复杂度可能接近暴力搜索的 $O(n \cdot k)$。对于维度大致大于 10-20 的情况，Ball 树或近似方法（LSH、HNSW）等其他结构通常表现更好。

让我们将一个简单的二维 k-d 树创建的划分可视化。我们可以将这些划分表示为散点图上的线。

````python
import plotly.graph_objects as go

def get_bounding_box(points: List[Point]) -> Tuple[Point, Point]:
    """计算一组点的边界框。"""
    all_points_np = np.array(points)
    min_coords = np.min(all_points_np, axis=0)
    max_coords = np.max(all_points_np, axis=0)
    return min_coords, max_coords

def plot_kdtree_partitions(node: Optional[Node], min_coords: Point, max_coords: Point, depth: int = 0, fig=None):
    """递归地向 Plotly 图形添加划分线。"""
    if node is None:
        return

    k = len(node.point)
    split_dim = node.split_dim # 使用存储的划分维度
    split_val = node.point[split_dim]

    # 绘制划分线/平面
    if split_dim == 0: # 垂直线
        fig.add_shape(type="line",
                      x0=split_val, y0=min_coords[1], x1=split_val, y1=max_coords[1],
                      line=dict(color="#adb5bd", width=1))
        # 使用更新后的边界递归处理子节点
        plot_kdtree_partitions(node.left, min_coords, np.array([split_val, max_coords[1]]), depth + 1, fig)
        plot_kdtree_partitions(node.right, np.array([split_val, min_coords[1]]), max_coords, depth + 1, fig)
    elif split_dim == 1: # 水平线
         fig.add_shape(type="line",
                      x0=min_coords[0], y0=split_val, x1=max_coords[0], y1=split_val,
                      line=dict(color="#adb5bd", width=1))
         # 使用更新后的边界递归处理子节点
         plot_kdtree_partitions(node.left, min_coords, np.array([max_coords[0], split_val]), depth + 1, fig)
         plot_kdtree_partitions(node.right, np.array([min_coords[0], split_val]), max_coords, depth + 1, fig)
    # 如果需要，可扩展到更高维度，尽管可视化将变得困难

# --- 绘图示例 ---
# 生成一些二维数据
np.random.seed(42)
points_list_2d = [np.random.rand(2) * 10 for _ in range(50)]

# 构建树
root_2d = build_kdtree(points_list_2d.copy())

# 创建散点图
points_np = np.array(points_list_2d)
fig = go.Figure(data=go.Scatter(x=points_np[:, 0], y=points_np[:, 1], mode='markers',
                               marker=dict(color='#228be6', size=8), name='数据点'))

# 添加查询点（可选）
query_pt = np.array([5, 5])
nearest_pt, _ = find_nearest_neighbor(root_2d, query_pt)
fig.add_trace(go.Scatter(x=[query_pt[0]], y=[query_pt[1]], mode='markers',
                         marker=dict(color='#fa5252', size=10, symbol='x'), name='查询点'))
if nearest_pt is not None:
    fig.add_trace(go.Scatter(x=[nearest_pt[0]], y=[nearest_pt[1]], mode='markers',
                             marker=dict(color='#e64980', size=10, symbol='star'), name='最近点'))

# 添加划分线
min_bound, max_bound = get_bounding_box(points_list_2d)
# 为可视化目的给边界添加少量边距
margin = 1.0
min_bound -= margin
max_bound += margin
plot_kdtree_partitions(root_2d, min_bound, max_bound, fig=fig)

fig.update_layout(title="k-d 树划分（二维示例）",
                  xaxis_title="维度 0", yaxis_title="维度 1",
                  xaxis=dict(range=[min_bound[0], max_bound[0]]),
                  yaxis=dict(range=[min_bound[1], max_bound[1]]),
                  width=700, height=600,
                  showlegend=True)

# 要显示图表（例如在 Jupyter 环境中）：
# fig.show()

# 要获取 JSON 表示：
# print(fig.to_json()) # 对于大量点/划分，这可能会非常长

# JSON 输出格式指南的简化示例（仅结构）：
# ```plotly
# {"layout": {"title": "k-d 树划分", "xaxis": {"title": "维度 0"}, "yaxis": {"title": "维度 1"}, "shapes": [{"type": "line", ...}, ...]}, "data": [{"type": "scatter", "mode": "markers", ...}, ...]}
# ```
````



![k-d 树划分（二维示例）](plots/2340-0.json)



> 散点图显示了 50 个随机生成的二维点、查询点（红色“x”）、其计算出的最近邻（粉色星形），以及由 k-d 树构建过程生成的前几条划分线（灰色）。垂直线沿维度 0 划分，水平线沿维度 1 划分。

这项实践练习演示了一个相对简单的递归数据结构如何能为像最近邻搜索这样的常见机器学习 (machine learning)任务提供很大的性能改进。通过自己实现它，您会对优化机器学习工作流程中涉及的算法选择有更透彻的理解，而不是仅仅依赖预构建的库函数。当您需要自定义算法或诊断空间数据问题中的性能问题时，理解这些结构是很有帮助的。

## 参考资料

- [An Algorithm for Finding Best Matches in Logarithmic Expected Time](https://dl.acm.org/doi/10.1145/355744.355745) — Jerome H. Friedman, Jon Louis Bentley, Raphael A. Finkel (1977)
  Journal: ACM Transactions on Mathematical Software; Publisher: Association for Computing Machinery; Volume: 3; Pages: 209-226; DOI: [10.1145/355744.355745](https://doi.org/10.1145/355744.355745)
  介绍k-d树的原始论文，为其结构和最近邻搜索奠定了理论基础。
- [Computational Geometry: Algorithms and Applications](https://link.springer.com/book/10.1007/978-3-540-77974-2) — Mark Berg, Otfried Cheong, Marc Kreveld, Mark Overmars (2008)
  Publisher: Springer; DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)
  一本关于几何算法的权威教材，全面讨论了k-d树及其在空间划分和最近邻查询中的应用。
- [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html) — scikit-learn developers (2023)
  scikit-learn最近邻算法的官方文档，涵盖k-d树、Ball树及其在机器学习场景中的特点。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本关于统计学习的教科书，讨论了k-最近邻算法及相关内容，并提及k-d树等高效搜索结构的作用。
