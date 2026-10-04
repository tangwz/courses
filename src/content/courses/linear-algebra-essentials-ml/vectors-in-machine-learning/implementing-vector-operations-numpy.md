---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "implementing-vector-operations-numpy"
sourceId: 1456
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/implementing-vector-operations-numpy"
title: "使用 NumPy 实现向量运算"
description: "使用 Python 中的 NumPy 库高效执行向量计算。"
order: 7
plots: []
sourceHash: "e80ace83a41fcc418d08a5c28f2906f81c47815f5566f7d7c6c1821d6a9030b0"
sourceCorrections: []
---

掌握向量 (vector)运算的数学原理是十分重要的，但要高效地执行这些计算，尤其是在机器学习 (machine learning)中常见的庞大数据集上，则需要专门的工具。Python 的 NumPy 库是数值计算的规范，并为处理向量（以及稍后将看到的矩阵）提供了优化且易于使用的函数。

NumPy 将向量表示为一维数组。让我们看看如何将我们讨论过的操作转换为 NumPy 代码。

### 使用 NumPy 创建向量 (vector)

首先，您需要导入 NumPy 库。标准做法是将其导入为 `np`。

```python
import numpy as np

# 从 Python 列表创建向量
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print("向量 A:", vector_a)
print("向量 B:", vector_b)
```

这会创建 NumPy 数组，它们是此情境下向量的主要数据结构。

### 向量 (vector)加法和减法

NumPy 允许您使用标准的 `+` 和 `-` 运算符执行逐元素的加法和减法，就像普通的数字一样。这比手动遍历列表元素要简洁和快速得多。

```python
# 向量加法
vector_sum = vector_a + vector_b
print("A + B:", vector_sum) # 输出: [5 7 9]

# 向量减法
vector_diff = vector_a - vector_b
print("A - B:", vector_diff) # 输出: [-3 -3 -3]
```

这些操作有效的前提是向量具有相同的维度，就像在数学定义中一样。如果您尝试对形状不兼容的向量执行加减运算，NumPy 将会报错。

### 标量乘法

用标量（一个数字）乘以向量 (vector)也很直接，使用 `*` 运算符。NumPy 会将乘法应用于向量的每个元素。

```python
scalar = 3
scalar_product = vector_a * scalar
print("标量 * A:", scalar_product) # 输出: [3 6 9]
```

### 点积

计算点积是常见的操作，经常用于衡量相似度或将一个向量 (vector)投影到另一个向量上。NumPy 提供了 `np.dot()` 函数或 `@` 运算符（Python 3.5 及更高版本）。

```python
# 使用 np.dot()
dot_product_np = np.dot(vector_a, vector_b)
print("点积 (np.dot):", dot_product_np) # 输出: 32 (1*4 + 2*5 + 3*6)

# 使用 @ 运算符
dot_product_operator = vector_a @ vector_b
print("点积 (@):", dot_product_operator) # 输出: 32
```

两种方法得到的结果相同。对于初学者来说，`np.dot()` 可能更明确，而 `@` 则提供更简洁的语法，尤其是在之后进行涉及矩阵的链式操作时。

### 计算向量 (vector)范数（长度）

NumPy 的线性代数模块 `np.linalg` 包含 `norm()` 函数，用于计算向量的长度。您可以使用 `ord` 参数 (parameter)指定范数的类型（L1、L2 等）。

```python
# L2 范数（欧几里得范数 - 默认）
l2_norm_a = np.linalg.norm(vector_a)
print(f"A 的 L2 范数: {l2_norm_a:.4f}") # 输出: ~3.7417

# L1 范数（曼哈顿范数）
l1_norm_a = np.linalg.norm(vector_a, ord=1)
print("A 的 L1 范数:", l1_norm_a) # 输出: 6.0 (1 + 2 + 3)

# 您可以验证 L2 范数计算: sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
print("手动 L2 检查:", np.sqrt(np.sum(vector_a**2))) # 输出: ~3.7417
```

`np.linalg.norm` 的默认值（未指定 `ord` 时）是 L2 范数，它是向量长度最常见的衡量方法，对应于欧几里得距离。

### 计算向量 (vector)间距离

两个向量 `a` 和 `b` 之间的距离通常计算为它们差值的范数，即 $||a - b||$。使用 NumPy，这是减法和范数函数的组合。

```python
# 欧几里得距离（差值的 L2 范数）
distance_l2 = np.linalg.norm(vector_a - vector_b)
print(f"A 和 B 之间的欧几里得距离: {distance_l2:.4f}") # 输出: ~5.1962

# 曼哈顿距离（差值的 L1 范数）
distance_l1 = np.linalg.norm(vector_a - vector_b, ord=1)
print("A 和 B 之间的曼哈顿距离:", distance_l1) # 输出: 9.0 (|-3| + |-3| + |-3|)
```

### 为何使用 NumPy？效率高！

尽管您*可以*使用标准 Python 列表和循环来实现这些操作，但 NumPy 操作要快得多。NumPy 函数是用 C 语言实现的，并针对数值任务进行了高度优化，在 Python 层面，它们一次性处理整个数组，而不是逐个元素地处理。这种效率在处理机器学习 (machine learning)中常见的庞大数据集时变得十分重要。

考虑一下对两个包含数百万个元素的向量 (vector)进行点积计算。纯 Python 循环的速度会比 `np.dot()` 慢几个数量级。

本节提供了使用 NumPy 实现基本向量操作的实用指南。熟练掌握这些工具对于高效处理机器学习算法中使用的数据表示形式非常必要。在下一章中，我们将把这些理念扩展到矩阵，矩阵使我们能够表示整个数据集并执行强大的变换。

## 参考资料

- [NumPy documentation](https://numpy.org/doc/stable/) — NumPy Developers (2024)
  了解 NumPy 数组及向量运算相关函数的必备参考资料。
- [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  实用指南，有效利用 NumPy 进行数值计算和数据处理，包含数组操作和性能细节。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  包含一个基础数学章节，涵盖线性代数概念，展示其在机器学习算法中的重要性。
