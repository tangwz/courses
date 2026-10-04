---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "matrix-operations"
lesson: "matrix-scalar-multiplication"
sourceId: 7815
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-3-matrix-operations/matrix-scalar-multiplication"
title: "矩阵与标量乘法"
description: "学习如何将整个矩阵乘以一个数。"
order: 2
plots: []
sourceHash: "a60f8f30aa2bb1ca4be7603bb89fba52af1c56dd7dab3852ab907639ff27ea08"
sourceCorrections: []
---

在对两个矩阵进行算术运算之后，下一个基本操作是用一个数对整个矩阵进行缩放。这称为矩阵与标量乘法，其工作原理与向量 (vector)与标量乘法完全相同。此操作涉及取一个标量（一个数），然后将矩阵中的每一个元素都乘以该标量。

设想你有一个以矩阵表示的灰度图像，其中每个元素代表一个像素的亮度值。如果你想让整个图像的亮度增加一倍，你只需将每个像素的值乘以2。这是一个关于矩阵与标量乘法的很好的类比；你正在对整个数值网格进行均匀缩放。

### 矩阵缩放的机制

如果我们有一个矩阵 $A$ 和一个标量 $c$，它们的乘积写作 $cA$。结果是一个与 $A$ 具有相同维度的新矩阵，其中每个元素是原始元素与该标量的乘积。

形式上，如果 $B = cA$，那么 $B$ 中第 $i$ 行第 $j$ 列的元素由以下式子给出：


$$
b_{ij} = c \times a_{ij}
$$


此操作是逐元素执行的，因此计算起来简单直接。下面的图示通过一个具体例子说明了此过程。

> 标量 `3` 乘以其右侧矩阵的每个元素，得到一个新矩阵，其中每个值都按3的倍数进行了缩放。

让我们演练上面例子的计算过程。给定矩阵 $A$ 和标量 $c=3$：


$$
A = \begin{bmatrix} 1 & 5 \\ 2 & -3 \end{bmatrix}
$$


乘积 $3A$ 的计算方式如下：


$$
3A = 3 \times \begin{bmatrix} 1 & 5 \\ 2 & -3 \end{bmatrix} = \begin{bmatrix} 3 \times 1 & 3 \times 5 \\ 3 \times 2 & 3 \times -3 \end{bmatrix} = \begin{bmatrix} 3 & 15 \\ 6 & -9 \end{bmatrix}
$$


### 它在机器学习 (machine learning)中有什么用处？

矩阵与标量乘法在机器学习算法中经常出现，通常以不显眼但作用明显的方式。

1. **特征缩放：** 在训练模型之前，通常会对数据集中的特征进行缩放。例如，如果你的数据集中某个特征以米为单位，你可以通过将整个列（一个向量 (vector)）乘以100将其转换为厘米。如果你想对*整个*数据集矩阵应用缩放因子，就会使用矩阵与标量乘法。
2. **学习率：** 在使用梯度下降 (gradient descent)训练神经网络 (neural network)和其他模型时，会使用一个称为“学习率”的主要参数 (parameter)。学习率是一个小的标量值（例如0.01），它控制着模型更新的步长。算法会计算一个梯度矩阵，该矩阵指示变化的方向，然后将此矩阵乘以学习率标量，以决定更新的幅度。这种缩放可以防止算法进行过大的更新，从而跳过最优解。

### 在NumPy中的实现

正如你可能想到的，在NumPy中执行此操作简单直观。你可以在NumPy数组（我们的矩阵）和标量之间使用标准乘法运算符 `*`。NumPy会自动为你处理逐元素乘法。

以下是在Python中执行我们示例中相同计算的方法：

```python
import numpy as np

# 定义我们的矩阵 A
A = np.array([
    [1, 5],
    [2, -3]
])

# 定义我们的标量 c
c = 3

# 执行标量乘法
B = c * A

print("原始矩阵 A：")
print(A)
print("\n标量 c:", c)
print("\n结果 c * A：")
print(B)
```

**输出：**

```
Original Matrix A:
[[ 1  5]
 [ 2 -3]]

Scalar c: 3

Result c * A:
[[ 3 15]
 [ 6 -9]]
```

NumPy代码的输出与我们的手动计算结果完全吻合。NumPy 能够无需显式循环，高效地在整个数组上执行这些操作，这一特性被称为向量 (vector)化 (vectorization)，它对编写简洁高效的Python数值代码很基本。在这个例子中，将单个标量应用于矩阵的每个元素，也是NumPy另一个相关特性——广播 (broadcasting)——的最简单情形：广播是NumPy让不同形状的数组（例如标量与矩阵，或向量与矩阵）之间也能进行运算的机制，其原理是将较小的数组"虚拟地"扩展以匹配较大数组的形状。
