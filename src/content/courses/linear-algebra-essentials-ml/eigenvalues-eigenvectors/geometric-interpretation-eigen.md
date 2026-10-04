---
course: "linear-algebra-essentials-ml"
chapter: "eigenvalues-eigenvectors"
lesson: "geometric-interpretation-eigen"
sourceId: 1513
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors/geometric-interpretation-eigen"
title: "几何解释"
description: "将特征向量看作线性变换后方向不变，仅按特征值缩放的方向。"
order: 2
plots: ["plots/1513-0.json", "plots/1513-1.json"]
sourceHash: "e4f969bb73c60ee8345f935a50795e0e614c9dbbaa1304e234aadb08990964dc"
sourceCorrections: []
---

定义 $Ax = \lambda x$ 起初可能看起来比较抽象，但它有着清晰的几何意义。可以将向量 (vector) $x$ 乘以矩阵 $A$ 看作是对 $x$ 进行线性变换。这种变换通常涉及旋转、剪切、缩放或它们的某种组合，会改变向量的方向和大小。

然而，特征向量有所不同。当对特征向量 $x$ 应用变换 $A$ 时，所得向量 $Ax$ 的方向与原始向量 $x$ 的方向**完全相同**（如果 $\lambda$ 为负，则方向完全相反）。变换仅仅将特征向量按因子 $\lambda$（对应的特征值）进行**缩放**。

想象一下，空间沿着某些轴被拉伸或收缩。完全沿着这些轴的向量就是特征向量。它们的方向不变；只是变长或变短。

我们来描绘一下。考虑一个由以下矩阵表示的简单缩放变换：
$A = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$
这个矩阵将 x 分量缩放 2 倍，将 y 分量缩放 0.5 倍。



![缩放矩阵变换](plots/1513-0.json)



> 原始向量（灰色点）被变换为新向量（红色点）。请注意，原始在 (1, 0) 的向量被变换到 (2, 0)。它的方向（沿着 x 轴）没有改变，仅仅是按 $\lambda = 2$ 的因子进行了缩放。类似地，在 (0, 1) 的向量变换到 (0, 0.5)，停留在 y 轴上，但按 $\lambda = 0.5$ 的因子进行了缩放。这些轴表示此变换的特征向量方向。像 (1, 1) 这样的向量变换到 (2, 0.5)，改变了它的方向（灰色箭头）。

特征向量不总是与标准的 x 轴和 y 轴对齐 (alignment)。考虑矩阵：
$B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$
这种变换同时涉及缩放和剪切。



![非轴对齐特征向量的变换](plots/1513-1.json)



> 这里，沿着 $y=x$ 这条线（例如 [1, 1]）的向量按 $\lambda = 3$ 的因子进行缩放（橙色箭头和虚线），沿着 $y=-x$ 这条线（例如 [1, -1]）的向量按 $\lambda = 1$ 的因子进行缩放（紫色箭头和虚线）。这些是矩阵 B 的特征向量方向。像 [1, 0] 这样的向量变换到 [2, 1]，改变了它的方向（灰色箭头）。

我们可以用一个图表来呈现这个核心思想：

> 如果 $x$ 是 $A$ 的特征向量，那么应用变换 $A$ 会得到一个向量 $Ax$，它与 $x$ 沿着穿过原点的同一条线。长度按因子 $\lambda$ 缩放，如果 $\lambda$ 为负，则方向反转 180 度。

当应用变换 $A$ 时，*非*特征向量的向量通常会改变它们的方向。只有这些特殊的特征向量方向保持不变（在缩放范围内）。

这种几何视角是根本的。它帮助我们理解，特征值和特征向量显示了线性变换纯粹作为缩放操作的“轴”。这就是为什么它们在主成分分析（PCA）等方法中如此重要，在这些方法中，我们寻找数据中方差最大（与特征值相关）的方向（特征向量）。理解这种几何行为为我们提供了直观的感受，说明为什么这些数学对象在数据分析和理解系统动态方面很有用。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本基础教材，对线性代数（包括特征值和特征向量）进行了全面且具几何直观的讲解。
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](http://vmlsbook.stanford.edu/) — Stephen Boyd, Lieven Vandenberghe (2018)
  Publisher: Cambridge University Press; Pages: 474
  一本实用线性代数入门书，强调与机器学习相关的应用，清晰解释了特征值等基本概念。
- [18.06SC Linear Algebra, Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/lecture-21-eigenvalues-and-eigenvectors/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  知名大学课程的讲义和视频讲座，从学术角度全面阐述特征值和特征向量，补充几何理解。
