---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "eigenvalues-and-eigenvectors"
lesson: "defining-eigenvalues-eigenvectors"
sourceId: 7828
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors/defining-eigenvalues-eigenvectors"
title: "定义特征值和特征向量"
description: "清晰地定义特征值和特征向量，它们是描述变换轴的独特标量和向量。"
order: 2
plots: []
sourceHash: "2c93fed3ae58ac5762f363b3ac4c07a4d9eda6875a2368cfd4b85af98a94bda4"
sourceCorrections: []
---

当一个矩阵作用于一个向量 (vector)时，它通常会改变向量的方向。但对于大多数方阵，某些非零向量是特别的。这种变换只会拉伸或收缩它们，保持它们原来的方向不变。这些特殊向量被称为**特征向量**，而它们被缩放的因子则是它们对应的**特征值**。

这种独特的关系用一个简单而有力的方程表示：


$$
Av = \lambda v
$$


这个方程说明，当矩阵 $A$ 作用于其特征向量 $v$ 时，结果等同于将 $v$ 乘以一个标量，即特征值 $\lambda$。

让我们分解每个组成部分：

- **$A$** 是表示线性变换的方阵。
- **$v$** 是**特征向量**，一个非零向量，其方向在变换中保持不变。
- **$\lambda$** (希腊字母 lambda) 是**特征值**，一个标量，它表示特征向量 $v$ 被拉伸、收缩或翻转的倍数。

本质上，特征向量定义了变换的“轴”。其他向量以复杂的方式旋转和缩放，而特征向量指向变换仅作为缩放操作的那些方向。

> 对于一般向量，矩阵变换会改变其方向。对于特征向量，变换只会沿着同一条线（其“特征空间”）对其进行缩放。

### 一个具体示例

让我们用数字来具体说明。考虑以下矩阵 $A$：


$$
A = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix}
$$


现在，我们来测试向量 (vector) $v = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ 是否是 $A$ 的一个特征向量。我们通过应用变换（将 $A$ 乘以 $v$）并查看结果是否是 $v$ 的一个缩放版本来完成此操作。

首先，计算方程的左侧，$Av$：


$$
Av = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 4(2) + (-2)(1) \\ 1(2) + 1(1) \end{pmatrix} = \begin{pmatrix} 8 - 2 \\ 2 + 1 \end{pmatrix} = \begin{pmatrix} 6 \\ 3 \end{pmatrix}
$$


变换的结果是向量 $\begin{pmatrix} 6 \\ 3 \end{pmatrix}$。现在，我们检查这个结果是否是我们原始向量 $v = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ 的倍数。

是否存在一个标量 $\lambda$，使得 $\begin{pmatrix} 6 \\ 3 \end{pmatrix} = \lambda \begin{pmatrix} 2 \\ 1 \end{pmatrix}$？

是的，存在。我们可以看出：


$$
\begin{pmatrix} 6 \\ 3 \end{pmatrix} = 3 \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$


因为我们找到了这样一个标量，我们已经确认 $v = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ 是矩阵 $A$ 的一个特征向量。对应的特征值是 $\lambda = 3$。变换 $A$ 将这个特定向量拉伸了 3 倍，而没有改变其方向。

### 特征值的解释

$\lambda$ 的值提供了关于变换如何作用于相应特征向量 (vector)的重要信息：

- 如果 **$\lambda > 1$**，特征向量被拉伸。
- 如果 **$0 < \lambda < 1$**，特征向量被收缩。
- 如果 **$\lambda = 1$**，特征向量在变换中保持不变。
- 如果 **$\lambda < 0$**，特征向量被翻转到相反方向，然后进行缩放。
- 如果 **$\lambda = 0$**，特征向量会坍缩到原点（零向量）。

### 非零条件

值得注意的是，特征向量 (vector)必须是**非零向量**。为什么？如果我们允许 $v$ 为零向量，方程 $A v = \lambda v$ 将变为 $A\mathbf{0} = \lambda\mathbf{0}$，简化后为 $\mathbf{0} = \mathbf{0}$。这个方程对任何矩阵 $A$ 和任何标量 $\lambda$ 都成立。它无法提供关于变换的有用信息，因此我们根据定义排除了零向量。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本广泛使用的线性代数入门教材，对特征值和特征向量进行了清晰的阐述。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2
  包含基础数学章节，涵盖线性代数概念（包括特征值和特征向量），对机器学习从业者有帮助。
- [18.06 Linear Algebra, Fall 2011](https://ocw.mit.edu/courses/18-06-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Journal: MIT OpenCourseWare
  提供一位备受尊敬的教授的视频讲座和课程材料，涵盖特征值和特征向量的基础知识。
