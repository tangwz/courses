---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "scalars-vectors-matrices-introduction"
lesson: "numpy-creating-vectors-matrices"
sourceId: 7807
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/numpy-creating-vectors-matrices"
title: "动手实践：使用 NumPy 创建向量和矩阵"
description: "一个关于使用 Python NumPy 库创建您的第一个向量和矩阵的实践教程。"
order: 6
plots: []
sourceHash: "3b1360d545321c1ee50f9bd4ecf65fa5a8fe0f44e57e0a860efddf28599f0f20"
sourceCorrections: []
---

创建标量、向量 (vector)和矩阵（这些是数据的基本构造）需要用到 Python 的 NumPy 库。NumPy 是 Python 中进行数值和科学计算的标准库，它提供了强大的工具来处理多维数组，而这正是向量和矩阵的本质。

如果您按照前面章节的设置指南操作，您应该已经安装了 NumPy。任何使用 NumPy 的脚本的第一步都是导入该库。标准做法是使用别名 `np` 导入它，这在几乎所有数据科学和机器学习 (machine learning)代码中都能看到。

```python
import numpy as np
```

这简单的一行代码让我们能够通过 `np` 前缀使用 NumPy 的所有函数。

### 创建向量 (vector)

正如我们所学，向量是一个有序的数字列表。在机器学习 (machine learning)中，向量通常代表一个数据点或一组特征。例如，我们可以用向量来代表一栋房子，包含三个特征：平方英尺数、卧室数量和浴室数量。

在 NumPy 中创建向量，我们使用 `np.array()` 函数并传入一个 Python 列表。

```python
# 代表一栋房子的特征向量：
# [平方英尺数, 卧室数量, 浴室数量]
house_vector = np.array([1500, 3, 2.5])

print(house_vector)
```

**输出：**

```
[1500.    3.    2.5]
```

这会创建一个一维 NumPy 数组。我们可以使用 `.shape` 属性来检查它的维度，这是理解数据结构的一个有用工具。

```python
print(house_vector.shape)
```

**输出：**

```
(3,)
```

输出 `(3,)` 表明这是一个包含三个元素的一维数组。依照惯例，这通常被称为**行向量**。

在线性代数中，行向量和列向量之间的区别值得注意。列向量被表示为具有多行但只有一列的矩阵。在 NumPy 中创建列向量，您必须使用嵌套列表，其中每个内部列表代表一行。

```python
# 创建一个列向量
house_column_vector = np.array([[1500], [3], [2.5]])

print(house_column_vector)
```

**输出：**

```
[[1500. ]
 [   3. ]
 [   2.5]]
```

现在，我们来检查它的形状。

```python
print(house_column_vector.shape)
```

**输出：**

```
(3, 1)
```

形状 `(3, 1)` 证实这是一个包含3行1列的二维对象。目前，我们大部分工作将使用一维数组，但了解如何创建列向量对于像矩阵-向量乘法这样的后续运算是必需的。

### 创建矩阵

矩阵以由行和列组成的二维网格形式组织数据。在机器学习 (machine learning)中，矩阵通常代表整个数据集，其中每一行是一个数据样本（一个向量 (vector)），每一列是所有样本中的一个特定特征。

让我们把房屋示例扩展为一个包含三栋房子的数据集。我们通过向 `np.array()` 传入一个列表的列表来创建一个矩阵。

```python
# 一个包含三栋房子的数据集
# 每一行是一栋房子，每一列是一个特征
housing_data_matrix = np.array([
    [1500, 3, 2.5],  # 房子 1
    [2100, 4, 3],    # 房子 2
    [1200, 2, 2]     # 房子 3
])

print(housing_data_matrix)
```

**输出：**

```
[[1500.    3.    2.5]
 [2100.    4.    3. ]
 [1200.    2.    2. ]]
```

检查我们矩阵的形状让我们清楚地了解它的结构。

```python
print(housing_data_matrix.shape)
```

**输出：**

```
(3, 3)
```

形状 `(3, 3)` 告诉我们这是一个包含3行3列的矩阵，完美匹配我们包含三栋房子、每栋房子有三个特征的数据集。这证实了我们之前建立的关联：矩阵实际上是向量的堆叠。

### 创建数组的有用函数

逐一输入每个数字并非总是实用。NumPy 提供了一些方便的函数来创建常见类型的矩阵。

#### 零和一矩阵

创建用占位值（例如全零或全一）初始化的数组经常有用。`np.zeros()` 和 `np.ones()` 函数用于此目的。您向它们提供一个元组来指定所需的形状。

```python
# 一个 2x4 的全零矩阵
zeros_matrix = np.zeros((2, 4))
print("零矩阵：")
print(zeros_matrix)

# 一个 3x2 的全一矩阵
ones_matrix = np.ones((3, 2))
print("\n一矩阵：")
print(ones_matrix)
```

**输出：**

```
Zeros Matrix:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Ones Matrix:
[[1. 1.]
 [1. 1.]
 [1. 1.]]
```

#### 具有范围值的数组

对于创建测试数据或序列，`np.arange()` 非常有用。它在给定区间内创建一个包含等距值的数组。它的运作方式很像 Python 的内置 `range()` 函数。

```python
# 创建一个包含从 0 到 9（不包括 9）的数字的向量
sequence_vector = np.arange(9)
print(sequence_vector)
```

**输出：**

```
[0 1 2 3 4 5 6 7 8]
```

当与 `.reshape()` 结合时，此函数变得更加强大，`.reshape()` 可以在不改变数据的情况下改变数组的形状。我们可以创建一个序列，然后立即将其构造成一个矩阵。

```python
# 创建一个包含从 0 到 8 的值的 3x3 矩阵
sequence_matrix = np.arange(9).reshape(3, 3)
print(sequence_matrix)
```

**输出：**

```
[[0 1 2]
 [3 4 5]
 [6 7 8]]
```

这种模式是生成用于测试算法的样本矩阵的快速且常见的方式。

借助这些工具，您现在可以将抽象的数学对象转换为具体的 NumPy 数组。您可以将单个数据点表示为向量 (vector)，将整个数据集表示为矩阵。在下一章中，我们将使用这些数组并开始对它们执行基本的线性代数运算。

## 参考资料

- [The absolute beginner's guide to NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html) — NumPy community (2023)
  涵盖NumPy数组创建、操作和形状等基本属性，对建立基础理解至关重要。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  提供了严谨的线性代数基础，包括向量和矩阵，直接适用于机器学习场景。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  提供使用NumPy进行数据操作的全面实践指导，包括数组创建和理解多维数组。
- [Introduction to Linear Algebra](https://math.mit.edu/linearalgebra/) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press
  一本经典教科书，介绍了核心线性代数概念，包括向量和矩阵的定义和属性。
