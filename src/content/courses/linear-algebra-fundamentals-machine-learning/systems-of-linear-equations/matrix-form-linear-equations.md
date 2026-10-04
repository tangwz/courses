---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "matrix-form-linear-equations"
sourceId: 7821
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/matrix-form-linear-equations"
title: "矩阵形式表示方程 (Ax = b)"
description: "学习如何将线性方程组转换为紧凑的矩阵-向量形式 Ax = b。"
order: 1
plots: []
sourceHash: "90eee8d1fbd1566aae64e2302cda8b8a8eae82cf4ee765f0ed05566a199e1955"
sourceCorrections: []
---

线性方程组是若干未知量之间关系的总和。例如，你可能有$x_1$和$x_2$这两个未知量，由以下两个方程关联：


$$
2x_1 + 3x_2 = 8
$$


$$
4x_1 - x_2 = 2
$$


求解此系统是指找到$x_1$和$x_2$的值，使这两个方程同时成立。虽然你可能曾手动使用代入法或消元法求解过这样的小型系统，但这种方法不适用于更大的规模。设想一个拥有数百个方程和变量的系统，这在机器学习 (machine learning)中很常见。我们需要一个更系统化且计算友好的方法。

线性代数在此提供了一种强大的方式来组织此问题。我们可以将任何线性方程组重写为简洁而优美的$Ax = b$形式。

### 拆解系统

我们再来看看我们的例子，并将其分解为三个主要部分：

1. **系数：** 它们是乘以变量的数字（如 2、3、4 和 -1）。
2. **变量：** 它们是我们希望求解的未知量（$x_1$和$x_2$）。
3. **常数：** 它们是结果，或等号右侧的值（8 和 2）。

核心思路是将这些组件类型分别归入各自的结构：系数放入矩阵$A$，变量放入向量 (vector)$x$，常数放入向量$b$。

### 组建矩阵和向量 (vector)

**1. 系数矩阵 `A`**

我们通过按方程中出现的相同布局排列系数来创建矩阵$A$。矩阵中的每一行对应一个方程，每一列对应一个变量。对于我们的系统，系数矩阵$A$是：


$$
A =
\begin{bmatrix}
2 & 3 \\
4 & -1
\end{bmatrix}
$$


第一行 `[2 3]` 包含第一个方程 ($2x_1 + 3x_2 = 8$) 的系数。第二行 `[4 -1]` 包含第二个方程 ($4x_1 - x_2 = 2$) 的系数。

**2. 变量向量 `x`**

接下来，我们将未知变量组合成一个列向量$x$。其顺序必须与矩阵$A$中列的顺序一致。由于$A$中的第一列对应于$x_1$，第二列对应于$x_2$，因此我们的向量$x$是：


$$
x =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$


**3. 常数向量 `b`**

最后，我们将方程右侧的常数收集到另一个列向量$b$中：


$$
b =
\begin{bmatrix}
8 \\
2
\end{bmatrix}
$$


以下图表展示了方程组如何转化为这三个不同的部分。

> 方程组的组件被组织成一个系数矩阵 A、一个变量向量 x 和一个常数向量 b。

### 验证矩阵方程

现在我们有了矩阵方程：


$$
\begin{bmatrix}
2 & 3 \\
4 & -1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
8 \\
2
\end{bmatrix}
$$


但我们怎么知道这与我们最初的系统相同？我们可以通过执行左侧的矩阵-向量 (vector)乘法来验证它，正如我们在第三章学到的。请记住，我们计算矩阵$A$的每一行与列向量$x$的点积：

- **第一行：** $(2 \cdot x_1) + (3 \cdot x_2)$
- **第二行：** $(4 \cdot x_1) + (-1 \cdot x_2)$

这种乘法会得到一个新的向量：


$$
\begin{bmatrix}
2x_1 + 3x_2 \\
4x_1 - x_2
\end{bmatrix}
$$


由于我们陈述 $Ax = b$，我们是在说：


$$
\begin{bmatrix}
2x_1 + 3x_2 \\
4x_1 - x_2
\end{bmatrix}
=
\begin{bmatrix}
8 \\
2
\end{bmatrix}
$$


要使两个向量相等，它们的对应元素必须相等。这让我们回到了最初的两个方程：$2x_1 + 3x_2 = 8$ 和 $4x_1 - x_2 = 2$。这证实了矩阵形式 $Ax = b$ 是我们原始系统完全有效且简洁的表示。

### 这种表示为何如此有用

将方程组转化为 $Ax = b$ 的形式不仅仅是一种符号上的技巧。它提供了多项显著的好处，尤其是在计算和机器学习 (machine learning)的背景下。

1. **简洁性与可扩展性：** 一个包含数百个方程和变量的系统可以像 $Ax = b$ 一样简单地写下来。其基础矩阵 $A$ 和向量 (vector) $x$ 和 $b$ 会大很多，但表示形式依然清晰。这使我们能够从更高的抽象层面思考问题。
2. **求解之道：** 这种形式提供了一种求解 $x$ 的方式。如果 $A$、$x$ 和 $b$ 只是数字，你会通过 $b$ 除以 $A$ 来求解 $x$。在线性代数中，等效操作是乘以*矩阵逆*，我们将在接下来的部分详细阐述。
3. **计算效率：** 现代数值计算库，如 NumPy，对矩阵和向量运算进行了高度优化。以这种形式表示问题使我们能够使用快速、预构建的函数来查找解，而不是编写缓慢的手动循环。求解 `Ax = b` 是这些库中的标准操作。

通过将方程组转换为这种通用矩阵形式，我们可以充分发挥线性代数和计算的全部能力来高效地找到解。在接下来的部分中，我们将学习求解向量$x$所需的工具。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  关于线性方程组及其矩阵表示的经典基础教材。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2
  第二章介绍了线性代数概念，包括Ax=b，重点关注它们与机器学习的相关性。
- [NumPy Reference Guide (numpy.linalg module)](https://numpy.org/doc/stable/reference/routines.linalg.html) — The NumPy Developers (2023)
  Publisher: NumPy
  NumPy线性代数模块的官方文档，说明了矩阵和向量操作的计算处理方式。
