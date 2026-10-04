---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "numpy-solving-linear-systems"
sourceId: 7826
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/numpy-solving-linear-systems"
title: "动手实践：使用NumPy求解方程组"
description: "使用NumPy函数计算矩阵逆并求解线性方程组的实用指南。"
order: 6
plots: ["plots/7826-0.json"]
sourceHash: "309e9591be5ec707bccf1c096f36553b53e6d0201b93f12a78c3c2067b764e82"
sourceCorrections: []
---

理论提供指引，但实际操作才能让你掌握具体情况。线性方程组可以用 $Ax = b$ 表示，而矩阵逆 $A^{-1}$ 可以帮助求得解向量 (vector) $x$。现在，你将使用NumPy（Python中进行数值计算的标准库）把这些数学知识转化为代码。

虽然计算逆矩阵是一种可行方法，但NumPy提供了一个更直接、高效且数值稳定的函数来直接求解 $x$。我们来看一个实际例子。

### 设置问题

设想你在购买水果。第一天，你买了2个苹果和3根香蕉，总共花费8美元。第二天，你买了4个苹果和1根香蕉，总共花费9美元。如果价格不变，一个苹果和一根香蕉的价格各是多少？

我们可以将其设定为一个包含两个未知数的二元线性方程组：
设 $x_1$ 为一个苹果的价格。
设 $x_2$ 为一根香蕉的价格。

该方程组为：


$$
2x_1 + 3x_2 = 8
$$


$$
4x_1 + 1x_2 = 9
$$


写成矩阵形式 $Ax = b$，则变为：


$$
\begin{pmatrix} 2 & 3 \\ 4 & 1 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 8 \\ 9 \end{pmatrix}
$$


我们的目的是求出向量 (vector) $x$，它包含一个苹果和一根香蕉的价格。

### 使用NumPy求解方程组

首先，我们导入NumPy并创建矩阵 `A` 和向量 (vector) `b`。

```python
import numpy as np

# 系数矩阵 (A)
A = np.array([
    [2, 3],
    [4, 1]
])

# 结果向量 (b)
b = np.array([8, 9])

print("Matrix A:\n", A)
print("\nVector b:\n", b)
```

现在，我们可以使用 `numpy.linalg.solve()` 函数。此函数专门设计用于求解 $Ax = b$ 形式的方程。它以矩阵 `A` 和向量 `b` 作为参数 (parameter)，并返回解向量 `x`。

```python
# 求解 x
x = np.linalg.solve(A, b)

print("\nSolution vector x:\n", x)
```

输出显示 $x_1 = 1.9$ 且 $x_2 = 1.4$。在我们的问题中，这意味着一个苹果的价格是1.90美元，一根香蕉的价格是1.40美元。

### 验证解

一个好的做法是验证我们的解是否正确。如果我们的 `x` 是正确的，那么将其乘以原始矩阵 `A` 应该得到原始结果向量 (vector) `b`。

我们来计算 $Ax$ 并查看它是否等于 $b$。

```python
# 通过计算 A @ x 来验证解
# @ 运算符执行矩阵乘法
verification = A @ x

print("Verification (A @ x):\n", verification)
print("\nOriginal vector b:\n", b)

# 检查验证结果是否接近 b
# 这对处理微小的浮点误差很有用
is_close = np.allclose(verification, b)
print("\nIs the solution correct?", is_close)
```

矩阵乘法 `A @ x` 的结果是 `[8. 9.]`，这与我们原始的 `b` 向量完全吻合。`np.allclose()` 函数是一种比较浮点数组的可靠方法，如果它们在很小的容差范围内逐元素相等，则返回 `True`。

### 逆矩阵法与 `solve()` 函数的比较

在前面章节中，我们讨论了通过求 $A$ 的逆并计算 $x = A^{-1}b$ 的方式来求解 $x$。你也可以在NumPy中这样做。

```python
# 找到 A 的逆
A_inv = np.linalg.inv(A)

# 使用逆计算 x
x_from_inverse = A_inv @ b

print("Solution from inverse:\n", x_from_inverse)
```

你得到相同的结果。那么 `np.linalg.solve()` 为什么存在呢？对于大型矩阵，`np.linalg.solve()` 速度更快且数值更稳定，与计算逆矩阵然后再进行矩阵乘法相比。`solve()` 函数采用更高级的分解技术，它避免了直接求逆的一些难题。通常来说，你应该优先使用 `np.linalg.solve()` 而非 `np.linalg.inv()`。

### 解的可视化

对于我们这样的二维系统，解表示两条线的交点。我们可以绘制这两条方程的图，以便直观地查看。

第一个方程 $2x_1 + 3x_2 = 8$ 可以改写为 $x_2 = (8 - 2x_1) / 3$。
第二个方程 $4x_1 + x_2 = 9$ 可以改写为 $x_2 = 9 - 4x_1$。

下面的图显示了这两条线及其交点，这就是我们的解。



![线性方程的交点](plots/7826-0.json)



> 方程组的解是两条线相交的那个点。这种几何解释有助于我们思考求解方程组意味着什么。

### 处理不可逆（奇异）方程组

如果一个方程组没有唯一解，会怎样？当矩阵 `A` 是奇异的（不可逆的）时就会发生这种情况，你已知晓，这意味着它的行列式为零。我们来看看NumPy如何处理这种情况。

考虑以下方程组：


$$
2x_1 + 3x_2 = 8
$$


$$
4x_1 + 6x_2 = 16
$$


第二个方程只是第一个方程的两倍。从几何角度看，它们是同一条线，因此有无限多个解。对应的矩阵是奇异的。

```python
# 奇异矩阵（第二行是第一行的2倍）
A_singular = np.array([
    [2, 3],
    [4, 6]
])

b_singular = np.array([8, 16])

# 我们尝试求解它
try:
    x_singular = np.linalg.solve(A_singular, b_singular)
except np.linalg.LinAlgError as e:
    print("Error:", e)
```

当你运行此代码时，NumPy会正确识别问题，并抛出带有“Singular matrix”消息的 `LinAlgError` 异常。这是NumPy在告知你，你定义的系统无法求解出一个唯一的解向量 (vector) `x`。这种实用的反馈非常有用，可以防止你的程序在遇到无法求解的系统时产生无意义的结果。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  全面介绍线性方程组、矩阵运算及其几何解释的理论基础教科书。
- [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) — NumPy Developers (2023)
  NumPy官方文档，详细介绍了`np.linalg.solve`函数的功能、用法和参数，用于求解线性系统。
- [SciPy Lecture Notes](https://scipy-lectures.org/) — Various Contributors (2022)
  Python科学计算实用指南，包含NumPy线性代数详细章节及求解系统时的性能考量。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖与机器学习相关的线性代数基础概念，包括线性系统求解和数值稳定性讨论。
