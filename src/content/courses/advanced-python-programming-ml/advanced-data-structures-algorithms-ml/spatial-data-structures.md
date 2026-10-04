---
course: "advanced-python-programming-ml"
chapter: "advanced-data-structures-algorithms-ml"
lesson: "spatial-data-structures"
sourceId: 2332
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/spatial-data-structures"
title: "空间数据结构（四叉树、八叉树）"
description: "在Python中实现四叉树或八叉树等空间索引结构，以实现机器学习中高效的空间查询。"
order: 4
plots: []
sourceHash: "f009b0d3efd74872ecf62a00c378ad77d75c47e231f6d5cf2957d7d8cf9b9a49"
sourceCorrections: []
---

许多机器学习 (machine learning)问题涉及具有固有空间特性的数据，例如地理坐标、图像中的像素位置或3D模拟中的点。高效处理此类数据通常需要专门设计用于处理多维空间的结构。对于查找特定半径内的所有点（范围查询）或定位查询点的最近邻等任务，标准列表甚至平衡二叉搜索树都变得效率低下，特别是当数据集增大时。

空间数据结构划分包含数据点的空间，从而在搜索过程中能够快速剪除不相关区域。本章的这一部分介绍两种基本的分层空间数据结构：用于二维空间的四叉树和用于三维空间的八叉树。

### 四叉树：组织二维空间

四叉树是一种树数据结构，其中每个内部节点恰好有四个子节点。它主要通过递归地将二维空间细分为四个象限或区域来划分空间。

#### 结构与构建

1. **根节点**: 代表包含所有数据点的整个二维边界框。
2. **细分**: 如果一个节点包含的点数超过预设容量（或如果它是一个需要进一步划分的内部节点），它所代表的空间将被分为四个相等的象限：西北 (NW)、东北 (NE)、西南 (SW) 和东南 (SE)。创建四个子节点，每个象限一个。
3. **点分布**: 父节点区域内的点根据其坐标分配到相应的子节点中。
4. **递归**: 这种细分过程对每个子节点递归地进行，直到没有节点包含的点数超过容量，或者达到最大树深度。未细分的节点是叶节点，通常存储落入其特定象限的点。

考虑将点插入到四叉树中。我们从根节点开始。如果根节点有子节点，我们判断该点属于哪个象限，并递归地尝试插入到该子节点中。如果节点是叶节点且容量未满，我们添加该点。如果叶节点已满，我们细分该节点，创建四个新的空子节点，并将之前在该叶节点中的所有点（包括新点）重新分配到相应的新子节点中。

```python
# 四叉树节点结构
class QuadTreeNode:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary # 表示矩形区域的对象
        self.capacity = capacity # 细分前的最大点数
        self.points = []         # 存储在该节点中的点（如果是叶节点）
        self.is_leaf = True
        self.nw = None           # 子节点
        self.ne = None
        self.sw = None
        self.se = None

    def subdivide(self):
        # 创建四个边界更小的子节点的逻辑
        self.is_leaf = False
        # ... create self.nw, self.ne, self.sw, self.se ...
        # 将 self.points 重新分配给子节点

    def insert(self, point):
        if not self.boundary.contains(point):
            return False # 点在该节点的边界之外

        if self.is_leaf:
            if len(self.points) < self.capacity:
                self.points.append(point)
                return True
            else:
                self.subdivide()
                # 细分后，尝试插入到正确的子节点
                # 继续处理非叶节点的情况

        # 不是叶节点，或刚刚被细分
        if self.nw.insert(point): return True
        if self.ne.insert(point): return True
        if self.sw.insert(point): return True
        if self.se.insert(point): return True
        
        # 如果边界检查正确，这不应该发生
        return False 
```

#### 查询

四叉树擅长空间查询：

- **范围查询**: 要查找查询矩形内的所有点，从根节点开始。如果查询矩形不与节点的边界相交，则剪除整个分支（包括其所有子节点）。如果它完全包含边界，则收集该子树中的所有点。否则，如果它部分重叠，则递归搜索相关子节点。
- **最近邻查询**: 查找距离查询点最近的点通常涉及一个优先队列。沿着树向下搜索包含查询点的象限。维护目前找到的最近点。当回溯到树的上层时，仅当其他分支的边界比当前最佳距离更接近查询点时，才进行检查。这能够剪除搜索空间的大部分区域。

> 一个四叉树结构划分二维空间。叶节点（蓝色）存储点，而内部节点（灰色）则进一步细分。

### 八叉树：扩展到三维空间

八叉树是四叉树在三维空间中的直接对应物。八叉树不将二维区域划分为四个象限，而是递归地将三维区域（通常是立方体或长方体）划分为八个八分体。

- **结构**: 每个内部节点有八个子节点，对应于八个子区域（例如，前下左、后上右）。
- **操作**: 插入、删除和搜索操作遵循与四叉树相同的原理，但在三维空间中操作。细分将立方区域分割成八个更小的立方体。
- **应用**: 八叉树在3D计算机图形（视锥体剔除、碰撞检测）、医学成像（分析MRI扫描等体数据）、科学模拟以及处理来自激光雷达或深度传感器的3D点云数据方面有价值，这在机器人和自动驾驶系统中越来越常见。

### 实现考量与机器学习 (machine learning)中的应用

在Python中实现四叉树或八叉树时：

- **边界表示**: 明确定义表示节点矩形（二维）或长方体（三维）边界的方式（例如，使用元组 `(min_x, min_y, max_x, max_y)` 或专门的类）。
- **点表示**: 点可以是简单的元组 `(x, y)` 或具有坐标属性的对象。
- **节点容量/深度**: 节点容量或最大深度的选择会影响性能和内存使用。较小的容量导致更深的树，但可能实现更快的剪枝，而较大的容量则产生较浅的树，每个叶节点可能存储更多的点。
- **库**: 虽然从零开始实现这些结构具有指导意义，但像 `Pyqtree`（用于四叉树）或专门的几何处理库可能提供优化的实现。

在机器学习中，这些结构特别适用于：

- **加速k-NN**: 对于低维空间数据（二维或三维），四叉树/八叉树可以显著加速最近邻搜索，优于与所有点进行暴力距离计算，尤其是在查询局部化时。它们划分空间，使算法能够快速舍弃远处区域的点。
- **空间索引**: 为大量地理数据点建立索引（例如，查找用户5公里范围内的所有商店）。
- **密度估计/聚类**: 像DBSCAN这样的算法可以受益于空间索引，以快速查找邻域半径($\epsilon$)内的点。
- **图形/模拟预处理**: 在将空间中的点或对象输入处理空间配置的机器学习模型之前，高效地组织它们。

### 局限性

四叉树和八叉树的主要局限是它们在高维空间 (high-dimensional space)中的性能下降（“维度灾难”）。随着维度数 ($d$) 增加，每个节点子节点数 ($2^d$) 呈指数增长。此外，空间划分的有效性降低，因为点在高维空间中倾向于变得等距，使得有效剪枝搜索分支变得更困难。对于更高维度的数据（通常 $d > 3$），像k-d树这样的结构（在“实现分层数据树”一节中介绍）或近似最近邻技术通常更合适。

理解四叉树和八叉树能帮助理解空间划分如何优化多维数据上的搜索和查询操作。虽然并非普遍适用于所有机器学习 (machine learning)数据，但它们在处理二维或三维的显式空间数据集时是不可或缺的工具，实现优于朴素方法的扩展性。

## 参考资料

- [Foundations of Multidimensional and Metric Data Structures](https://www.elsevier.com/books/foundations-of-multidimensional-and-metric-data-structures/samet/978-0-12-369446-1) — Hanan Samet (2006)
  Publisher: Morgan Kaufmann
  一部涵盖Quadtree和Octree等空间数据结构理论与设计的学术著作。
- [Computational Geometry: Algorithms and Applications](https://link.springer.com/book/10.1007/978-3-540-77974-2) — Mark Berg, Otfried Cheong, Marc Kreveld, Mark Overmars (2008)
  Publisher: Springer; DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)
  一本标准的教材，提供计算几何中的基本算法和应用，包括空间数据组织方法。
- [\`scipy.spatial.KDTree\` and \`scipy.spatial.cKDTree\` Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) — SciPy Developers (2023)
  SciPy中k-d树实现的官方文档，提供Python中高效的空间查询，与本文讨论的结构相关。
