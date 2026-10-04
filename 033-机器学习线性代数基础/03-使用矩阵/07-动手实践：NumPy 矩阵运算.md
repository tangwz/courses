# 动手实践：NumPy 矩阵运算

来源：[原文](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-3-matrix-operations/numpy-matrix-operations-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理论给出规则，而计算才是实际操作的环节。在 Python 代码中使用 NumPy 库执行矩阵运算，能够实际应用其底层原理。NumPy 是 Python 科学计算的根本，因为它提供高度优化的数组处理工具，这些工具非常适合表示向量 (vector)和矩阵。

在本节中，你将学会如何进行我们讨论过的所有核心矩阵运算。掌握这些 NumPy 函数是必要的步骤，无论是从头开始实现机器学习 (machine learning)模型，还是为 Scikit-learn 和 TensorFlow 等常用库妥善准备数据。

### 配置环境

首先，我们需要导入 NumPy 库。标准做法（你也应该一直遵循）是将其导入并使用别名 `np`。

```python
import numpy as np
```

### 创建 NumPy 矩阵

我们通过调用 `np.array()` 并传入一个嵌套列表来创建矩阵，其中每个内部列表代表一行。让我们定义一些矩阵和一个向量 (vector)，以用于我们的例子。

```python
# 一个 2x3 矩阵
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

# 一个 3x2 矩阵
B = np.array([[7, 8], 
              [9, 10], 
              [11, 12]])

# 用于逐元素运算的 2x3 矩阵
C = np.array([[2, 4, 6], 
              [8, 10, 12]])

# 一个大小为 2 的向量（一维数组）
v = np.array([100, 200])

print("矩阵 A (2x3):\n", A)
print("\n矩阵 B (3x2):\n", B)
print("\n矩阵 C (2x3):\n", C)
print("\n向量 v:\n", v)
```

### 逐元素运算

逐元素运算是对矩阵间对应元素执行的。为了使这些运算生效，涉及的矩阵必须具有完全相同的维度。

#### 矩阵加法和减法

你可以使用标准运算符 `+` 和 `-` 对同形状的矩阵进行加减。NumPy 会自动处理逐元素计算。

```python
# 矩阵加法 (A 和 C 都是 2x3)
add_result = A + C
print("A + C:\n", add_result)

# 输出:
# A + C:
#  [[ 3  6  9]
#  [12 15 18]]
```

```python
# 矩阵减法
subtract_result = C - A
print("C - A:\n", subtract_result)

# 输出:
# C - A:
#  [[1 2 3]
#  [4 5 6]]
```

尝试将 `A`（一个 2x3 矩阵）和 `B`（一个 3x2 矩阵）相加会导致 `ValueError`，因为它们的形状不兼容。

#### 矩阵-标量乘法

要将矩阵的每个元素乘以一个数（标量），请使用 `*` 运算符。

```python
# 将矩阵 A 乘以标量 10
scalar_mult_result = A * 10
print("A * 10:\n", scalar_mult_result)

# 输出:
# A * 10:
#  [[10 20 30]
#  [40 50 60]]
```

### 矩阵乘法

矩阵乘法是线性代数中一项非常重要的运算。它不是逐元素运算，并且对矩阵维度有特别的规定。在 NumPy 中，你应该使用 `@` 运算符进行矩阵乘法。

#### 矩阵-向量 (vector)乘法

矩阵乘以向量会改变向量。让我们使用矩阵 `A` 来转换向量 `v`。这是行不通的，因为内维度不匹配：`A` 的形状是 (2, 3)，`v` 的形状是 (2,)。我们需要 `A` 的列数等于 `v` 的元素数量。

让我们使用一个兼容的 2x2 矩阵 `D` 来乘以 `v`。

```python
# 一个 2x2 矩阵
D = np.array([[1, 2],
              [3, 4]])

# 将 D (2x2) 乘以 v (2,)
# 结果形状将是 (2,)
mv_result = D @ v
print("矩阵 D:\n", D)
print("\n向量 v:\n", v)
print("\nD @ v:\n", mv_result)

# 输出:
# 矩阵 D:
#  [[1 2]
#  [3 4]]
#
# 向量 v:
#  [100 200]
#
# D @ v:
#  [ 500 1100]
```

进行的计算是第一个元素为 $(1 \cdot 100 + 2 \cdot 200)$，第二个元素为 $(3 \cdot 100 + 4 \cdot 200)$。

#### 矩阵-矩阵乘法

要将两个矩阵相乘，第一个矩阵的列数必须等于第二个矩阵的行数。例如，我们可以将 `A` (2x3) 乘以 `B` (3x2)。结果矩阵的形状将是 (2x2)。

```python
# 将 A (2x3) 乘以 B (3x2)
# 结果形状将是 (2x2)
mm_result = A @ B
print("A @ B:\n", mm_result)

# 输出:
# A @ B:
#  [[ 58  64]
#  [139 154]]
```

> 注意：初学者常犯的一个错误是使用 `*` 运算符进行矩阵乘法。在 NumPy 中，`*` 执行逐元素乘法。要使两个矩阵进行逐元素乘法，它们必须具有相同的形状。`@` 运算符是在 Python 3.5 中专门为矩阵乘法引入的，是推荐的方法。

### 矩阵转置

转置矩阵是将其沿主对角线翻转，将行变成列，反之亦然。在 NumPy 中，你可以使用 `.T` 属性获取矩阵的转置。

```python
# A (2x3) 的转置将是一个 3x2 矩阵
A_transpose = A.T
print("原始矩阵 A (2x3):\n", A)
print("\nA 的转置 (3x2):\n", A_transpose)

# 输出:
# 原始矩阵 A (2x3):
#  [[1 2 3]
#  [4 5 6]]
#
# A 的转置 (3x2):
#  [[1 4]
#  [2 5]
#  [3 6]]
```

### 创建特殊类型的矩阵

NumPy 提供了便捷的函数来创建线性代数中常用的矩阵。

#### 单位矩阵

单位矩阵是一个对角线上全是 1，其余元素都是 0 的方阵。你可以使用 `np.identity()` 创建一个。

```python
# 创建一个 3x3 单位矩阵
I = np.identity(3)
print("3x3 单位矩阵:\n", I)

# 输出:
# 3x3 单位矩阵:
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

将任何矩阵乘以一个兼容的单位矩阵，原矩阵保持不变。

```python
# A 是 2x3，I 必须是 3x3 才能让 A @ I 生效
I_3x3 = np.identity(3)
result = A @ I_3x3

print("A @ I:\n", result)
print("\n结果与 A 相同吗?", np.array_equal(result, A))

# 输出:
# A @ I:
#  [[1. 2. 3.]
#  [4. 5. 6.]]
#
# 结果与 A 相同吗? True
```

#### 对角矩阵

对角矩阵只有主对角线上有非零值。你可以使用 `np.diag()` 从列表或一维值数组中创建一个对角矩阵。

```python
# 从值列表中创建一个对角矩阵
diag_matrix = np.diag([2, 5, 8])
print("对角矩阵:\n", diag_matrix)

# 输出:
# 对角矩阵:
#  [[2 0 0]
#  [0 5 0]
#  [0 0 8]]
```

你现在已经练习了 NumPy 中所有基本的矩阵运算。这些函数构成了你将反复用于数据处理和构建机器学习 (machine learning)算法的工具集。在下一章中，我们将看到如何使用这些运算来解线性方程组。

## 参考资料

- [NumPy documentation: Array creation and operations](https://numpy.org/doc/stable/user/basics.html) — NumPy Developers (2024)
  理解NumPy数组对象、创建及基本操作（如逐元素运算、矩阵乘法和转置）的重要资源。
- [Python for Data Analysis](https://www.porchlightbooks.com/product/python-for-data-analysis-data-wrangling-with-pandas-numpy-and-jupyter--wes-mckinney) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  在Python数据处理和科学计算的背景下，提供了NumPy数组创建和矩阵运算的实用示例及详细解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2
  第2章简明地概述了与机器学习直接相关的线性代数基础知识，强化了矩阵运算的理论基础。

---

[上一节](06-%E7%89%B9%E6%AE%8A%E7%9F%A9%E9%98%B5%E7%B1%BB%E5%9E%8B.md) · [下一节](../04-%E7%BA%BF%E6%80%A7%E6%96%B9%E7%A8%8B%E7%BB%84/01-%E7%9F%A9%E9%98%B5%E5%BD%A2%E5%BC%8F%E8%A1%A8%E7%A4%BA%E6%96%B9%E7%A8%8B%20%28Ax%20%3D%20b%29.md)
