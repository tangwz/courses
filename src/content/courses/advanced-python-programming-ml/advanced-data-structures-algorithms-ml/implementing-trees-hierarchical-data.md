---
course: "advanced-python-programming-ml"
chapter: "advanced-data-structures-algorithms-ml"
lesson: "implementing-trees-hierarchical-data"
sourceId: 2324
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/implementing-trees-hierarchical-data"
title: "实现分层数据的树结构"
description: "使用 Python 实现各种树结构（二叉树、k-d 树）及其在机器学习中的应用（例如决策树、最近邻算法）。"
order: 1
plots: []
sourceHash: "59d24678cb64bda2b249728438ab6f35524b83f42bdeb3b4510a5aa2f553b008"
sourceCorrections: []
---

树提供了一种有效的方式来表示分层关系和划分数据空间，超越了列表或数组的线性组织。在机器学习 (machine learning)中，这项功能经常被运用，尤其是在涉及高效搜索、索引或决策过程表示的任务中。尽管标准库通常提供优化的树实现，但了解其底层结构和运作方式对于定制算法或分析性能瓶颈很有帮助。

机器学习中常见的树结构包括其实现和应用。主要关注以二叉树作为基本结构，以及 k-d 树在实际空间索引中的应用。

## 表示分层数据

许多数据集都表现出固有的分层结构。例如文件系统、组织架构图，甚至是某些算法中特征空间的递归划分。树为这些情况提供了一种自然的数据结构。树由通过边连接的节点组成，其中有一个指定的根节点，通常没有循环。每个节点可以有零个或多个子节点。

## 二叉树

最简单的形式是二叉树，其中每个节点最多有两个子节点，通常称为左子节点和右子节点。

### 基本实现

在 Python 中实现二叉树节点的一种简单方法是使用类：

```python
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

# 使用示例：
# 表示一个简单的表达式树，例如 (3 + 5) * 2
root = Node('*')
root.left = Node('+')
root.left.left = Node(3)
root.left.right = Node(5)
root.right = Node(2)

print(f"根节点值: {root.value}")
print(f"左子节点值: {root.left.value}")
print(f"右子节点值: {root.right.value}")
```

尽管是基础，但基本的二叉树本身不常直接用于复杂的机器学习 (machine learning)任务。然而，像二叉搜索树 (BST) 和决策树的结构都是建立在这个思路之上。BST 维护一个排序属性（左子节点值 < 节点值 < 右子节点值），这使得搜索效率较高（平均为 $O(\log n)$），但在不平衡数据上其性能会下降，并且与其它结构相比，它们在处理高维机器学习特征空间时较不常见。

## K维树 (k-d 树)

机器学习 (machine learning)中一种特别有用的树结构是 k-d 树。它是一种空间划分数据结构，用于在 $k$ 维空间中组织点，这使其在最近邻搜索中非常有效。

### 构建

k-d 树通过递归划分数据点来构建。在树的每个级别，划分沿着 $k$ 个维度中的一个进行。一种常见的策略是在每个级别循环使用维度。

1. **选择轴**: 选择一个维度（轴）进行划分（例如，循环 $x, y, z, \dots$ 或选择方差最大的轴）。
2. **选择枢轴**: 找到沿所选轴的中位数点。该点成为节点。
3. **划分**: 将剩余的点分为两个子集：沿该轴小于枢轴值的点进入左子树，大于或等于的点进入右子树。
4. **递归**: 对左右子集重复步骤 1-3，直到子集为空或包含单个点。

### 可视化

考虑划分一个二维空间 ($k=2$)。第一次划分可能沿着 x 轴，使用 x 值的中间值。随后生成的左右子树将沿着 y 轴，使用它们各自 y 值的中间值进行划分，依此类推。

> 2D k-d 树生成的前几次空间划分的可视化。根节点垂直（X轴）划分空间，随后的节点则水平（Y轴）划分各自的子空间。

### 应用：最近邻搜索

k-d 树在机器学习 (machine learning)中的主要用途是加速 $k$ 最近邻 ($k$-NN) 查询。相比于将查询点与数据集中的每个点进行比较（一种 $O(N \cdot k)$ 复杂度的暴力方法，其中 $N$ 是点的数量），k-d 树能实现更快的搜索，在有利条件下（例如，低维度、合理分布的数据）平均复杂度通常为 $O(k \log N)$。

搜索算法通过遍历树来执行：

1. 沿着树向下移动，根据查询点相对于每个节点划分维度的坐标选择路径，直到到达叶节点。
2. 计算查询点到叶节点处点（或多个点）的距离。这是当前最近邻的最佳估计。
3. 回溯到树的上方。在每个节点，检查另一个子树是否可能包含比当前最佳估计更近的点。这涉及检查以查询点为中心、半径等于当前最佳距离的超球体是否与另一个子树的边界框相交。
4. 如果另一个子树可能包含更近的点，则遍历该子树，如果找到更近的点则更新最佳估计。
5. 继续回溯直到到达根节点。

### 维度考量

尽管在低到中等维度（例如 $k < 20$）下效率高，但随着维度增加，k-d 树的性能优势会减弱。这通常被称为“维度灾难”。在高维空间 (high-dimensional space)中，划分效果变差，搜索算法可能需要查看树的很大一部分，接近暴力搜索的复杂度。对于非常高维的数据，像球树这样的替代结构可能更合适。

### Python 实现概览

实现 k-d 树需要仔细处理递归、中位数查找（可能使用 `introselect` 等算法）以及节点信息管理（点坐标、划分维度、左/右子节点）。像 `scipy.spatial.KDTree` 或 Scikit-learn 的 `sklearn.neighbors.KDTree` 这样的库提供了高度优化的实现。然而，理解其核心逻辑很有帮助：

```python
import numpy as np

class KDNode:
    def __init__(self, point=None, split_dim=None, left=None, right=None):
        self.point = point          # 此节点的数据点
        self.split_dim = split_dim  # 用于划分的维度
        self.left = left            # 左子节点
        self.right = right          # 右子节点

def build_kdtree(points, depth=0):
    n = len(points)
    if n <= 0:
        return None

    # 根据第一个点确定维度
    k = len(points[0])
    # 循环维度
    axis = depth % k

    # 根据当前轴对点进行排序并选择中位数
    # 使用中位数的中位数或类似方法优于完全排序
    sorted_points = sorted(points, key=lambda point: point[axis])
    median_idx = n // 2

    # 创建节点并递归构建子树
    node = KDNode(point=sorted_points[median_idx], split_dim=axis)
    node.left = build_kdtree(sorted_points[:median_idx], depth + 1)
    node.right = build_kdtree(sorted_points[median_idx+1:], depth + 1)

    return node

# 示例（使用简单的二维点）
points_2d = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
kdtree_root = build_kdtree(points_2d)

# 注意：这是一个简化的构建函数。生产实现
# 需要高效的中位数查找和边缘情况处理。
# 实际使用还需要搜索功能。

if kdtree_root:
    print(f"根点: {kdtree_root.point}, 划分维度: {kdtree_root.split_dim}")
    if kdtree_root.left:
        print(f"左子点: {kdtree_root.left.point}")
    if kdtree_root.right:
        print(f"右子点: {kdtree_root.right.point}")
```

“这个简化示例展示了递归结构和轴循环。实现中会优化中位数查找并处理沿划分轴的潜在重复点。”

## 总结

树结构为分层组织数据和划分特征空间提供了必要方法。尽管基本的二叉树展示了主要思想，但 k-d 树提供了一种实用且广泛运用的方法，用于加速机器学习 (machine learning)中的最近邻搜索，特别是在低到中等维度下。理解 k-d 树如何划分特征空间以及搜索算法如何剪枝分支，对于掌握 $k$-NN 等算法的性能特点以及辨识其效率可能因高维度而降低的情形很有帮助。这些认识有助于在实现或优化涉及空间查询的机器学习算法时选择合适的数据结构。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press
  全面介绍了基本数据结构（包括二叉树和二叉搜索树）及其算法分析。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://statweb.stanford.edu/~tibs/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 745
  在机器学习背景下讨论了K-近邻算法和K-d树等高效空间索引技术。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser (2013)
  Publisher: Wiley
  提供了各种树结构（包括二叉树基础知识）的Python实现和解释。
- [scipy.spatial.KDTree - SciPy v1.13.0 Reference Guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) — SciPy Developers (2024)
  Python中高效K-d树实现的官方文档，详细说明了其在空间查询中的用法、参数和性能特征。
