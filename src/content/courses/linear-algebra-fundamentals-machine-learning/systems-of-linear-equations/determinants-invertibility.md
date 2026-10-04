---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "determinants-invertibility"
sourceId: 7824
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/determinants-invertibility"
title: "行列式与可逆性"
description: "了解如何计算行列式并用它来判断矩阵是否可逆。"
order: 4
plots: []
sourceHash: "183db3dee891920fc51e7973d5d1c83c382fe3a4f4a1e35d5b20ff9ec36c0da4"
sourceCorrections: []
---

如果矩阵 `A` 有逆矩阵，那么我们可以解出 $Ax = b$ 中的 $x$。这就引出了一个要事：我们怎么知道一个矩阵是否有逆矩阵呢？并非所有矩阵都可逆，尝试用不可逆矩阵求解方程组可能会出问题。我们用来检查此事的工具就是**行列式**。

行列式是一个从方阵元素计算得出的单一标量值。这个数字能反映矩阵及其所代表的线性变换的许多信息。

### 行列式的几何视角

明白行列式的一种有效方法是思考它的几何意义。当矩阵 `A` 乘以一个向量 (vector)时，它会变换这个向量。如果我们把同样的变换应用于构成一个形状（比如简单的单位正方形）的所有向量，那么形状本身也会被变换。

2x2 矩阵的行列式表示该变换对区域的**缩放因子**。

例如，二维空间中的单位正方形面积为 1。如果我们对其施加一个矩阵变换，并且得到的形状（通常是平行四边形）面积为 5，那么该变换矩阵的行列式就是 5。如果面积是 0.5，行列式就是 0.5。

最需要注意的情况是行列式为**零**时。行列式为零表示变换将原始形状压缩成了没有面积的东西，比如一条线或一个点。这是一种会丢失维度的破坏性变换。

> 非零行列式表明变换会缩放空间，这是可逆的。零行列式表明变换会压缩空间，这是不可逆的。

如果一个变换将二维正方形压缩成一维直线，你就丢失了信息。没有办法“解压缩”那条线并完美地恢复原始正方形。这种信息丢失就是零行列式的矩阵不可逆的原因。

### 计算行列式

对于一个简单的 2x2 矩阵，行列式的计算公式很直接。给定一个矩阵 $A$：


$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$


行列式，通常写作 `det(A)` 或 $|A|$，计算方式如下：


$$
\text{行列式}(A) = ad - bc
$$


让我们看一个例子：


$$
A =
\begin{bmatrix}
4 & 1 \\
2 & 3
\end{bmatrix}
$$


行列式是：


$$
\text{行列式}(A) = (4 \times 3) - (1 \times 2) = 12 - 2 = 10
$$


因为行列式是 10（非零），我们知道这个矩阵是可逆的。

对于大于 2x2 的矩阵，计算会变得更繁琐，但道理相同。实际应用中，你几乎总是会使用像 NumPy 这样的库来计算行列式，而不是手动计算。

### 可逆性规则

这引出了连接行列式与逆矩阵的核心规则：

> 一个方阵可逆当且仅当其行列式非零。

这是线性代数中的一个基本性质。在尝试找到矩阵的逆矩阵或求解依赖于逆矩阵的方程组之前，你可以先计算行列式。

- 如果 `det(A) ≠ 0`，则存在逆矩阵，且 $Ax = b$ 有唯一解。
- 如果 `det(A) = 0`，则逆矩阵不存在。方程组可能无解或有无数解，但不会有唯一解。

### 奇异矩阵与非奇异矩阵

这个性质引出了一些你将常遇到的正式术语：

- **非奇异矩阵**：行列式**非零**的方阵。非奇异矩阵**可逆**。
- **奇异矩阵**：行列式为**零**的方阵。奇异矩阵**不可逆**（也称为退化矩阵）。

因此，“这个矩阵有逆矩阵吗？”这个问题与“这个矩阵是非奇异的吗？”是相同的。通过计算行列式，你可以回答这两个问题。

## 参考资料

- [Introduction to Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/syllabus/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本备受推崇的教材和课程，以其对行列式及其在理解矩阵变换和可逆性中的作用的直观几何解释而闻名。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 30-55
  第2章回顾了线性代数的基础知识，包括行列式和矩阵可逆性，这些是机器学习算法的基础。
- [numpy.linalg.det](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html) — NumPy Developers (2023)
  NumPy库计算矩阵行列式的官方文档，NumPy是Python中用于机器学习数值计算的常用工具。
