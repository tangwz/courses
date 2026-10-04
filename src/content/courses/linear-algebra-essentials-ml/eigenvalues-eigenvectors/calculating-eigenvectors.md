---
course: "linear-algebra-essentials-ml"
chapter: "eigenvalues-eigenvectors"
lesson: "calculating-eigenvectors"
sourceId: 1516
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors/calculating-eigenvectors"
title: "计算特征向量"
description: "通过求解每个特征值的系统 (A - \\u03bbI)x = 0 来找到特征向量。"
order: 4
plots: []
sourceHash: "ac410897b19e0ce0e06c65613c45f0f2da3ae8c3fb85e438f7e7da4dd12a6afc"
sourceCorrections: []
---

特征方程，$det(A - \lambda I) = 0$，用于找到给定方阵 $A$ 的特征值 ($\lambda$)。特征值表示特征向量 (vector)经矩阵 $A$ 变换后被缩放的倍数。下面将为每个特征值找到对应的实际特征向量 ($x$)。

我们从矩阵、其特征值及其对应的特征向量之间的基本定义开始：
$Ax = \lambda x$

我们的目的是找到满足特定特征值 $\lambda$ 的这个方程的非零向量 $x$。我们可以重新排列此方程，将所有项移到一边：
$Ax - \lambda x = 0$

继续下一步，我们需要提取向量 $x$。我们可以引入单位矩阵 $I$（与 $A$ 具有相同维度）而不改变方程，因为 $Ix = x$：
$Ax - \lambda I x = 0$

现在，我们可以提取 $x$：
$(A - \lambda I)x = 0$

这个方程是一个**齐次线性方程组**。它看起来很像 $Mx = 0$ 的形式，其中 $M = (A - \lambda I)$。我们正在寻找该方程组的非零解 $x$。

为什么我们能保证找到非零解？回顾一下，我们通过求解 $det(A - \lambda I) = 0$ 找到了特征值 $\lambda$。行列式为零的矩阵称为**奇异矩阵**。奇异矩阵不可逆，并且当 $M$ 是奇异矩阵时，$Mx = 0$ 系统的一个重要性质是它有无穷多个非平凡（非零）解。这些 $x$ 的非零解正是与特征值 $\lambda$ 相关的特征向量。

### 计算过程

因此，对于您计算出的**每一个**特征值 $\lambda$：

1. **构造矩阵：** 构造矩阵 $M = A - \lambda I$。这涉及从原始矩阵 $A$ 的每个对角线元素中减去特征值 $\lambda$。
2. **建立系统：** 写出齐次方程组 $Mx = 0$，即 $(A - \lambda I)x = 0$。
3. **求解系统：** 找到向量 (vector) $x$ 的非零解。此处的标准方法是对矩阵 $M = A - \lambda I$ 应用**高斯消元法**（行化简）。您实际上是在寻找矩阵 $(A - \lambda I)$ 的**零空间**。
4. **确定特征向量：** 解通常会涉及一个或多个自由变量。这意味着不存在唯一的解向量 $x$。相反，这些解构成一个子空间，称为该特征值 $\lambda$ 的**特征空间**。此特征空间中的任何非零向量都是一个有效的特征向量。通常，我们会选择一个简单的代表向量（通常通过将自由变量设置为一个方便的值，例如 1）。

### 计算示例

让我们举例说明一个简单的 2x2 矩阵。考虑以下矩阵：
$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$

假设我们已经使用特征方程找到了特征值。假设其中一个特征值是 $\lambda = 5$。现在我们找到对应于 $\lambda = 5$ 的特征向量 (vector)。

1. **构造矩阵 $(A - \lambda I)$：**
   $A - \lambda I = A - 5I = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} - 5 \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} - \begin{bmatrix} 5 & 0 \\ 0 & 5 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}$
2. **建立系统 $(A - \lambda I)x = 0$：**
   令 $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$。该系统为：
   $\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
   这转化为以下方程：
   $-x_1 + x_2 = 0$
   $2x_1 - 2x_2 = 0$
3. **求解系统：**
   注意到第二个方程只是第一个方程的 -2 倍。它们是线性相关的，这是预料之中的，因为矩阵 $(A - 5I)$ 是奇异的。从第一个方程中，我们得到：
   $x_2 = x_1$
   变量 $x_1$ 是一个自由变量。我们可以为其选择任何值（除了 0，因为特征向量必须是非零的），$x_2$ 将由此确定。
4. **确定一个特征向量：**
   让我们为自由变量选择一个简单值，比如 $x_1 = 1$。那么 $x_2 = 1$。
   因此，对应于特征值 $\lambda = 5$ 的一个特征向量是：
   $x = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
   该向量的任何非零标量倍数，例如 $\begin{bmatrix} 2 \\ 2 \end{bmatrix}$ 或 $\begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}$，也都是 $\lambda = 5$ 的有效特征向量。所有此类向量的集合构成了 $\lambda=5$ 的特征空间。

对于为矩阵 $A$ 找到的任何其他特征值，您都会重复此过程。

尽管手动进行高斯消元对于较小的矩阵很有启发性，但对于机器学习 (machine learning)应用中遇到的大型矩阵，您将依赖 NumPy 等数值库，这些库具有高效函数可以直接计算特征值和特征向量。然而，理解这个底层过程，能够对这些函数的作用以及如何解释其结果提供有益的见解。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本广泛使用的教材，涵盖了特征值、特征向量、零空间和高斯消元等线性代数基础主题。
- [Mathematics for Machine Learning](https://doi.org/10.1017/9781108679934) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679934](https://doi.org/10.1017/9781108679934)
  提供了机器学习的数学基础，其中包含关于线性代数概念（包括特征值和特征向量）的专门章节。
- [Linear Algebra (18.06SC)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  麻省理工学院的开放式在线课程，提供讲解和材料，说明如何通过行约简方法计算特征向量。
- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — NumPy Developers (2024)
  Publisher: NumPy
  NumPy函数官方文档，用于计算特征值和相应的特征向量，与数值实现相关。
