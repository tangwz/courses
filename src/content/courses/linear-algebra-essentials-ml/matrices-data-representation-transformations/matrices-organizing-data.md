---
course: "linear-algebra-essentials-ml"
chapter: "matrices-data-representation-transformations"
lesson: "matrices-organizing-data"
sourceId: 1461
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-2-matrices-data-representation-transformations/matrices-organizing-data"
title: "矩阵用于组织数据"
description: "了解矩阵如何构建数据集以及如何在机器学习中表示向量的集合。"
order: 1
plots: []
sourceHash: "17c8e2eac74c4606dccf42674d8e26ef600846bded2d5c0b6f3c3b848ded92b0"
sourceCorrections: []
---

在上一章中，我们看到向量 (vector)如何作为基本构成要素，常在机器学习 (machine learning)中代表单个数据点或一次观测的一组特征。例如，一个向量可以包含一张图像的像素值，一篇文档的词频，或一套房屋的特征（如面积、卧室数量）。

"然而，机器学习任务几乎总是处理*多个*数据点，而不仅仅是一个。我们需要一种有效的方法来组织和处理这些集合。这就是矩阵的作用。"

矩阵是由数字组成的矩形网格，按行和列排列。如果一个矩阵有 $m$ 行和 $n$ 列，其维度就是 $m 	imes n$。矩阵是表示多个数据点的有力工具，并且可以看作是向量的扩展形式。向量本身就是一个只有一列（列向量）或一行的矩阵（行向量）。

### 从数据集到矩阵

在机器学习 (machine learning)中，矩阵最常见的应用是表示整个数据集。想象一个存储在表格中的典型数据集，例如电子表格或CSV文件：

| 面积 (平方英尺) | 卧室数 | 价格 (千美元) |
| --- | --- | --- |
| 1500 | 3 | 300 |
| 1200 | 2 | 250 |
| 1800 | 4 | 380 |
| 1350 | 3 | 290 |

我们可以直接将这种表格数据转换为矩阵。通常，矩阵的每一行对应一个*样本*或*数据点*（如一套房屋），每一列对应一个特定的*特征*或*属性*（如面积、卧室数量或价格）。

对于上述房屋数据，如果我们将“面积”和“卧室数”作为输入特征 ($X$) 来预测“价格” ($y$)，我们可以将这些特征表示为一个矩阵：


$$
X = \begin{bmatrix}
1500 & 3 \\
1200 & 2 \\
1800 & 4 \\
1350 & 3
\end{bmatrix}
$$


这个矩阵 $X$ 的维度是 $4 \times 2$，因为它有4个样本（房屋）和2个特征。每一行，例如 $[1500, 3]$，都是一个样本的特征向量 (vector)。目标变量“价格”通常会作为单独的向量存储：


$$
y = \begin{bmatrix}
300 \\
250 \\
380 \\
290
\end{bmatrix}
$$


这个矩阵 $X$ 通常被称为**特征矩阵**或**设计矩阵**。它是许多机器学习算法中使用的标准结构。

> 该图示说明了数据集表格（其中行是样本，列是特征）如何通常被表示为特征矩阵 $X$。矩阵中的每一行都是单个样本的特征向量。

### 其他数据类型表示为矩阵

矩阵不限于简单的表格数据。考虑机器学习 (machine learning)中常见的其他数据类型：

- **图像：** 灰度图像可以直接表示为矩阵，其中每个元素对应一个像素的强度值。一个 $28 \times 28$ 像素的灰度图像成为一个 $28 \times 28$ 矩阵。对于彩色图像（例如RGB），您可以使用三个独立的矩阵（一个用于红色，一个用于绿色，一个用于蓝色）或将它们堆叠成一个称为张量的三维结构（张量将矩阵的表示方式扩展到更多维度）。
- **时间序列：** 随时间推移的一系列测量值有时可以结构化为矩阵，也许是通过创建重叠的观测窗口，其中每一行代表一个窗口。
- **图/网络：** 邻接矩阵表示图中节点之间的连接，其中 $A_{ij}$ 表示节点 $i$ 和节点 $j$ 之间是否存在边。

### 为何使用矩阵？

将数据组织成矩阵提供了几个重要优势：

1. **紧凑表示：** 矩阵提供了一种简洁的方式来存储和引用大量结构化数据。
2. **标准化形式：** 这种结构被普遍理解，并构成了许多算法和软件包的基础。
3. **运算基础：** 很快我们就会看到，定义像矩阵乘法这样的运算，使我们能够同时对整个数据集高效地执行复杂的数据转换（如旋转或缩放）和计算（如求解方程组）。
4. **计算效率：** 像NumPy这样的库针对矩阵上的数值运算进行了高度优化（在NumPy中常被称为`ndarrays`或N维数组）。使用NumPy对整个矩阵执行操作比在Python中循环处理单个数据点要快得多。这种“向量 (vector)化”对于机器学习 (machine learning)的性能非常重要。

通过将数据点集合表示为矩阵中的行（或有时是列），我们创建了一种理想的结构化形式，便于数学操作和计算处理，为我们接下来将介绍的线性变换和方程组求解提供了方法。我们将大量依赖NumPy在Python中创建和使用这些矩阵表示。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  提供矩阵的基础知识、定义以及与数据组织相关的基本数学运算。
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — Stephen Boyd and Lieven Vandenberghe (2018)
  Publisher: Cambridge University Press
  涵盖线性代数概念，直接应用于数据表示和统计机器学习问题。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书开头章节回顾了机器学习所需的基本数学概念，包括如何使用矩阵组织和处理数据。
- [NumPy Documentation](https://numpy.org/doc/stable/) — NumPy Developers (2023)
  NumPy官方文档，提供在Python中创建和操作N维数组（矩阵）的实用信息，用于高效计算。
