---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "fundamental-vector-operations"
sourceId: 1446
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/fundamental-vector-operations"
title: "基本向量运算"
description: "了解向量加法、减法和标量乘法及其几何解释。"
order: 2
plots: []
sourceHash: "db08f8f7e09741201f06560650ec993d3c23ddc0cdb135d36b96effefde43d29"
sourceCorrections: []
---

既然我们理解了向量 (vector)如何表示数据，让我们来看看可以对它们进行的基本运算。这些运算构成了许多机器学习 (machine learning)算法的组成部分。我们将介绍加法、减法和标量乘法，研究它们的数学定义和几何解释。

### 向量 (vector)加法

两个向量相加很简单：你将它们对应的元素相加即可。如果你有两个相同维度的向量 $v$ 和 $w$ (例如，在 $\mathbb{R}^n$ 中)，它们的和 $v + w$ 是一个新向量，其中每个元素是 $v$ 和 $w$ 中相同位置元素的和。

Mathematically, if $v = [v_1, v_2, ..., v_n]$ and $w = [w_1, w_2, ..., w_n]$, then:
$v + w = [v_1 + w_1, v_2 + w_2, ..., v_n + w_n]$

**几何解释：** 从几何上看，向量加法可以用“首尾相连”法或平行四边形法则来表示。设想将向量 $w$ 的起点放在向量 $v$ 的终点处。得到的向量 $v + w$ 从原点 (向量 $v$ 的起点) 开始，到向量 $w$ 的新终点结束。另一种方式是，如果你从原点开始，用 $v$ 和 $w$ 构成一个平行四边形，那么从原点出发的对角线就表示和 $v + w$。

让我们用两个二维向量 $v = [2, 1]$ 和 $w = [1, 3]$ 来显示这一点。它们的和是 $v + w = [2+1, 1+3] = [3, 4]$。

> 向量 $v$ (蓝色) 通过首尾相连法与 $w$ (绿色，从 $v$ 的终点开始的实线) 相加。得到的向量 $v+w$ (橙色) 从原点指向最终终点。虚线绿色向量表示从原点开始的 $w$。

### 向量 (vector)减法

与加法类似，向量减法涉及减去对应的元素。对于相同维度的向量 $v$ 和 $w$：
$v - w = [v_1 - w_1, v_2 - w_2, ..., v_n - w_n]$

**几何解释：** 向量减法 $v - w$ 可以看作是将向量 $w$ 的负向量加到 $v$ 上，即 $v + (-w)$。向量 $-w$ 的大小与 $w$ 相同，但方向相反。从几何上看，当 $v$ 和 $w$ 都从原点开始时，向量 $v - w$ 是从 $w$ 的终点开始，到 $v$ 的终点结束的向量。这个向量表示从 $w$ 到 $v$ 的位移。

考虑 $v = [4, 2]$ 和 $w = [1, 3]$。那么 $v - w = [4-1, 2-3] = [3, -1]$。

> 向量 $v$ (蓝色) 和 $w$ (绿色) 都从原点开始。向量 $v-w$ (橙色) 表示从 $w$ 的终点到 $v$ 的终点的位移。

### 标量乘法

标量乘法是指将向量 (vector)乘以一个单一的数 (标量)。如果标量是负数，这个运算会改变向量的大小，并可能反转其方向。如果 $c$ 是一个标量，而 $v = [v_1, v_2, ..., v_n]$ 是一个向量：
$c v = [c v_1, c v_2, ..., c v_n]$

**几何解释：** 将向量 $v$ 乘以标量 $c$，如果 $c > 0$，则得到的新向量与 $v$ 方向相同；如果 $c < 0$，则方向相反。得到向量的大小 (长度) 是原始向量 $v$ 大小的 $|c|$ 倍。如果 $c = 0$，结果是零向量。

例如，如果 $v = [2, 1]$ 和 $c = 2.5$：
$2.5 v = [2.5 \times 2, 2.5 \times 1] = [5, 2.5]$
这个新向量与 $v$ 方向相同，但长度是 $v$ 的 2.5 倍。

如果 $c = -1$：
$-1 v = [-1 \times 2, -1 \times 1] = [-2, -1]$
这个向量与 $v$ 长度相同，但方向相反。

> 原始向量 $v$ (蓝色)，按 2.5 倍缩放 (橙色)，以及按 -1 倍缩放 (紫色)。缩放会拉伸或收缩向量，并可以反转其方向。

这些基本运算，即加法、减法和标量乘法，是重要的工具。它们使我们能够组合和处理数据的向量表示，构成了我们后续将遇到的更复杂算法和转换的根本，例如计算质心、更新模型参数 (parameter)或投影数据。在接下来的部分中，我们将看到如何使用 Python 的 NumPy 库高效地执行这些操作。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press; Pages: Chapter 1
  一本广受推崇的教科书，提供了线性代数概念的基础理解，包括向量运算及其几何表示的清晰解释。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; Pages: Chapter 2
  本书为机器学习提供了坚实的数学基础，从与机器学习实践者高度相关的角度讲解了向量运算。
- [Linear Algebra (MIT OpenCourseWare Course 18.06)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Gilbert Strang (2010)
  Publisher: MIT OpenCourseWare
  一门经典的在线课程，提供视频讲座和讲义，为基本向量运算提供直观的几何解释。
