---
course: "linear-algebra-essentials-ml"
chapter: "matrices-data-representation-transformations"
lesson: "matrices-as-linear-transformations"
sourceId: 1467
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-2-matrices-data-representation-transformations/matrices-as-linear-transformations"
title: "矩阵作为线性变换"
description: "将矩阵乘法解释为执行缩放、旋转和剪切等线性变换的一种方式。"
order: 4
plots: ["plots/1467-0.json", "plots/1467-1.json", "plots/1467-2.json"]
sourceHash: "7dc8a041472e9b711d9fc4d02aa53baef5c4a4feddb0da7950b27e28a7fd6bc2"
sourceCorrections: []
---

在上一节中，我们学习了矩阵乘法纯粹的计算过程。现在，我们来看看它的几何意义。将矩阵乘以向量 (vector)不只是符号运算；这是一种在空间中*变换*该向量（或它所代表的点）的方式。这种变换方式在机器学习 (machine learning)中非常重要，用于从改变坐标系到处理数据表示等任务。

可以将矩阵 $A$ 看作是一个算子或函数，它接收一个输入向量 $x$，并通过乘法生成一个输出向量 $y$：$y = Ax$。这种运算称为**线性变换**。

是什么使变换具有“线性”特征？它必须对任意向量 $u, v$ 和任意标量 $c$ 满足两个性质：

1. $A(u + v) = Au + Av$（保持加法）
2. $A(cu) = c(Au)$（保持标量乘法）

矩阵乘法本身就满足这些条件。从几何角度看，线性变换保持原点（零向量映射到零向量），并保持网格线平行且间隔均匀，尽管间隔和方向可能会变。它们可以拉伸、收缩、旋转、反射或剪切空间，但不会使之弯曲。

我们来查看一些由作用于二维向量的 2x2 矩阵表示的常见线性变换。我们将可视化它们如何影响单位正方形的顶点：$(0,0), (1,0), (1,1), (0,1)$。

### 缩放

缩放变换沿坐标轴拉伸或收缩空间。它由对角矩阵表示。

考虑矩阵 $S = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$。我们将其应用于向量 (vector) $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$：

$Sx = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2x_1 \\ 0.5x_2 \end{bmatrix}$

这种变换使任何向量的 $x_1$ 分量加倍，并使 $x_2$ 分量减半。将其应用于单位正方形的顶点：

- $(0,0) \rightarrow (0, 0)$
- $(1,0) \rightarrow (2, 0)$
- $(1,1) \rightarrow (2, 0.5)$
- $(0,1) \rightarrow (0, 0.5)$



![缩放变换](plots/1467-0.json)



> 缩放变换 [[2, 0], [0, 0.5]] 应用于单位正方形。正方形水平拉伸，垂直压缩。

### 旋转

旋转变换使向量 (vector)绕原点旋转一个特定角度 $\theta$。标准二维旋转矩阵是：

$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

我们旋转 $\theta = 90^\circ$（$\cos 90^\circ = 0, \sin 90^\circ = 1$）。矩阵变为 $R_{90} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$。

将其应用于单位正方形的顶点：

- $(0,0) \rightarrow (0, 0)$
- $(1,0) \rightarrow (0, 1)$
- $(1,1) \rightarrow (-1, 1)$
- $(0,1) \rightarrow (-1, 0)$



![90度旋转](plots/1467-1.json)



> 旋转变换 [[0, -1], [1, 0]] 应用于单位正方形，使其逆时针旋转90度。

### 剪切

剪切变换使空间倾斜，相当于使层之间相互滑动。水平剪切通常由形如 $H = \begin{bmatrix} 1 & s \\ 0 & 1 \end{bmatrix}$ 的矩阵表示，其中 $s$ 是剪切因子。

我们使用 $s = 1$，因此 $H = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$。将其应用于向量 (vector) $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$：

$Hx = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 + x_2 \\ x_2 \end{bmatrix}$

这种变换将 $x_2$ 坐标加到 $x_1$ 坐标，根据点的高度水平移动。将其应用于单位正方形的顶点：

- $(0,0) \rightarrow (0, 0)$
- $(1,0) \rightarrow (1, 0)$
- $(1,1) \rightarrow (1+1, 1) = (2, 1)$
- $(0,1) \rightarrow (0+1, 1) = (1, 1)$



![水平剪切](plots/1467-2.json)



> 水平剪切变换 [[1, 1], [0, 1]] 应用于单位正方形。顶部边缘向右移动。

### 组合变换

如果我们先应用一个变换，然后再应用另一个变换，会发生什么？例如，先旋转90度（$R_{90}$），然后缩放（$S$）？我们顺序应用这些变换：

1. $x' = R_{90} x$
2. $y = S x' = S (R_{90} x)$

由于矩阵乘法满足结合律，这等同于 $y = (S R_{90}) x$。组合变换由单个变换矩阵的乘积表示。

**注意：** 矩阵乘法通常不满足交换律（$AB \neq BA$）。这意味着应用变换的顺序很重要。先缩放后旋转通常会产生与先旋转后缩放不同的结果。对应于*首先*应用的变换的矩阵出现在乘积的右侧。

### 与机器学习 (machine learning)的关联

将矩阵理解为线性变换在机器学习中意义重大：

- **特征空间变换：** 诸如主成分分析（PCA）之类的技术通过应用旋转和缩放（由矩阵表示）来为数据寻找新的坐标系（基），以沿轴线最大化方差。特征分解和奇异值分解（SVD）将在后续讨论，它们都源于此。
- **数据增强：** 在计算机视觉中，旋转、缩放或剪切图像（以矩阵或像素值张量表示）可以创建现有数据的变体，以提高模型的泛化能力。
- **神经网络 (neural network)层：** 神经网络中的一些层执行仿射变换（$Ax + b$），其中矩阵乘法 $Ax$ 是线性变换组成部分。

在后续章节和部分中，我们将看到这些几何解释如何与求解线性系统、理解向量 (vector)空间以及执行在整个机器学习中都有用到的强大矩阵分解联系起来。请记住，NumPy 提供高效工具（如 `@` 运算符或 `np.dot()`）来执行这些矩阵乘法，从而有效地应用我们讨论过的变换。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本标准教材，涵盖线性代数基础，包括矩阵变换的几何解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第2章简要介绍了线性代数，包括线性变换，适合机器学习从业者。
