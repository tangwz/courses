---
course: "linear-algebra-essentials-ml"
chapter: "matrix-decompositions-machine-learning"
lesson: "geometric-interpretation-svd"
sourceId: 1529
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/geometric-interpretation-svd"
title: "SVD 的几何含义"
description: "将 SVD 想象为旋转、缩放和再次旋转的序列。"
order: 3
plots: ["plots/1529-0.json"]
sourceHash: "101a8f118cd1efeb7359985e6c7dde5cba256af72cb15c7177f1c760eda4e83e"
sourceCorrections: []
---

奇异值分解（SVD）的公式 $A = U\Sigma V^T$ 尽管看起来抽象，但它描述了一个基本的几何过程。理解这种几何性质有助于直观把握 SVD 为何在降维和理解线性变换等应用中如此有效。

设想一个由 $m \times n$ 矩阵 $A$ 表示的线性变换。当我们对向量 (vector) $x$ 应用此变换得到 $y = Ax$ 时，SVD 告诉我们这个变换可以分解为三个不同的几何步骤：

1. **旋转/反射 ($V^T$):** 矩阵 $V^T$（$V$ 的转置）是一个正交矩阵。当应用于输入向量 $x$ 时，$V^T x$ 在输入空间 $\mathbb{R}^n$ 中执行旋转并可能进行反射。它不改变向量的长度或它们之间的夹角。可以将此步骤看作是让输入空间沿着一组特殊的正交方向对齐 (alignment)，这些方向由 $V$ 的列（右奇异向量）给出。这些向量 $v_1, v_2, \dots, v_n$ 构成输入空间的一个标准正交基。$V^T$ 本质上是旋转空间，使得这些主输入方向与标准坐标轴对齐。
2. **缩放 ($\Sigma$):** 矩阵 $\Sigma$ 是一个 $m \times n$ 矩形对角矩阵。其对角线元素是奇异值 $\sigma_1, \sigma_2, \dots, \sigma_r$（其中 $r$ 是 $A$ 的秩），所有其他元素均为零。此矩阵缩放旋转后的向量 $V^T x$ 的坐标。具体来说，它将第 $i$ 个坐标（对应于 $v_i$ 的方向）按奇异值 $\sigma_i$ 缩放。对应于零奇异值的方向有效地压缩为零。此步骤沿着新对齐的轴线拉伸或收缩空间。
3. **旋转/反射 ($U$):** 矩阵 $U$ 也是一个正交矩阵（$m \times m$）。这最后一步将缩放后的向量 $\Sigma V^T x$ 在输出空间 $\mathbb{R}^m$ 中执行另一次旋转并可能进行反射。$U$ 的列 $u_1, u_2, \dots, u_m$（左奇异向量）构成输出空间的一个标准正交基。此步骤将缩放后的向量从轴对齐的方向（步骤 2 之后）旋转到它们在输出空间中的最终位置，使它们与由 $U$ 的列定义的主输出方向对齐。

### 变换的可视化

设想将变换 $A$ 应用于 2D 单位圆（或 3D 单位球）上的所有点。SVD 几何上分解变换过程如下：

1. **输入：** 从以原点为中心的单位圆开始。
2. **应用 $V^T$：** 这会旋转单位圆。由于是刚性变换，结果仍是单位圆，仅仅是方向可能不同。由右奇异向量 (vector) $v_1, v_2$ 定义的轴旋转以对齐 (alignment)标准轴 $e_1, e_2$。
3. **应用 $\Sigma$：** 这会沿着标准轴线按比例 $\sigma_1$ 和 $\sigma_2$ 缩放旋转后的圆。圆变成一个椭圆，其长轴和短轴与标准坐标轴对齐。半轴的长度为 $\sigma_1$ 和 $\sigma_2$。
4. **应用 $U$：** 这会旋转椭圆而不改变其形状。最终结果是一个椭圆，其轴与左奇异向量 $u_1, u_2$ 对齐。

SVD 本质上告诉我们，任何线性变换 $A$ 将输入空间中的标准正交基向量（$V$ 的列）映射到输出空间中的正交向量（通过奇异值 $\sigma_i$ 缩放的 $U$ 的列）。也就是说，$Av_i = \sigma_i u_i$，其中 $i = 1, \dots, r$。



![矩阵 A 对单位圆的几何作用](plots/1529-0.json)



> 由矩阵 $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ 定义的变换将单位圆（灰色）映射为椭圆（红色）。SVD 将这一整体变换分解为一系列的旋转 ($V^T$)、轴对齐的缩放 ($\Sigma$) 和另一次旋转 ($U$)。

这种几何视角对于理解使用 SVD 进行降维尤其有帮助。奇异值 $\sigma_i$ 量化 (quantization)了每个主方向的重要性。较大的奇异值对应于数据（或变换）具有最大方差或“扩散”的方向。通过仅保留对应于最大奇异值的分量，我们保留了变换最重要的几何特征，同时可能丢弃与小奇异值相关的维度（这些维度可能代表噪声或不重要的变化）。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第二章线性代数内容简要介绍了SVD，并将其几何解释与机器学习应用联系起来。
- [Linear Algebra (Course 18.06)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Prof. Gilbert Strang (2010)
  Publisher: MIT OpenCourseWare
  第29和30讲涵盖了SVD及其几何解释，提供了直观的理解。
