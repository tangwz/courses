---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "singular-vs-nonsingular-matrices"
sourceId: 7825
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/singular-vs-nonsingular-matrices"
title: "奇异矩阵与非奇异矩阵"
description: "区分奇异矩阵（不可逆）与非奇异矩阵（可逆）。"
order: 5
plots: []
sourceHash: "163b3f6ba6edfc47ef7574057204c056fae70372a2c332fe589b1e4e30bc4b4c"
sourceCorrections: []
---

在上一节中，我们知晓矩阵的行列式是一个单独的数值，其中包含许多信息。现在，我们将了解这个数值如何帮助我们把矩阵分为两种不同但重要的类型：非奇异和奇异。这种分类直接表明了我们能从线性方程组中得到的解的类型。

### 非奇异矩阵：获得唯一解的途径

具有逆矩阵的方阵被称为**非奇异**或**可逆**矩阵。识别它的最直接方式是看它的行列式。

如果矩阵 $A$ 是非奇异的，当其行列式不为零时。


$$
det(A) \neq 0
$$


当我们系统 $Ax = b$ 中的矩阵 $A$ 是非奇异时，这表示向量 (vector) $x$ 存在一个单一的、唯一的解。这是理想情况。因为逆矩阵 $A^{-1}$ 存在，我们可以用它来清晰地解此系统：


$$
x = A^{-1}b
$$


从几何角度看，非奇异矩阵变换空间时不会损失任何维度。例如，一个二维非奇异矩阵可能将一个正方形旋转、拉伸或剪切成一个平行四边形，但它始终会保留为一个具有正面积的二维形状。这种变换可以完全逆转，这正是逆矩阵的作用。

> 非奇异矩阵将一个正方形变换为一个平行四边形。面积会改变，但它不会塌缩成一条线。这种变换是可逆的。

### 奇异矩阵：无唯一解的情况

另一方面，不具有逆矩阵的方阵被称为**奇异**或**不可逆**矩阵。

如果矩阵 $A$ 是奇异的，当其行列式恰好为零时。


$$
det(A) = 0
$$


如果 $Ax = b$ 中的矩阵 $A$ 是奇异的，我们使用逆矩阵的方法会立即失效，因为 $A^{-1}$ 不存在。这表明方程系统本身存在问题。这表示系统要么**无解**，要么**有无数个解**。它永远不会有一个唯一解。

从几何角度看，奇异矩阵将空间压缩到更低的维度。例如，一个二维奇异矩阵会将一个二维向量 (vector)平面塌缩到一条单线甚至一个单点上。这种操作是不可逆的。一旦你将一个二维形状压平为一条线，就没有信息能告知你如何将其“复原”回原始形状了。

> 奇异矩阵将一个二维正方形塌缩为一条一维直线。信息会丢失，且变换不可逆。

### 这种区分在机器学习 (machine learning)中的重要性

在线性回归等机器学习算法中，奇异矩阵常表示输入数据存在问题。如果两个特征存在冗余，就可能出现这种情况。例如，如果你有一个表示摄氏温度的特征和另一个表示华氏温度的特征，它们就完全线性相关。表示你数据的矩阵将是奇异的，且解决线性回归的标准方法会失效，因为它依赖于矩阵逆。识别奇异矩阵能帮助你诊断数据集中存在的此类问题。

### 区别总结

此表格提供非奇异矩阵和奇异矩阵性质的快速参考。

| 特征 | 非奇异矩阵 | 奇异矩阵 |
| --- | --- | --- |
| **可逆性？** | 是 | 否 |
| **行列式** | 不为零 ($det(A) \neq 0$) | 为零 ($det(A) = 0$) |
| **$Ax = b$ 的解** | 一个唯一解 | 无解或无数个解 |
| **几何效应** | 保持维度 | 维度塌缩 |
| **线性独立性** | 列（和行）是独立的 | 列（和行）是相关的 |

### 在 Python 中检验奇异性

我们可以使用 NumPy 轻松判断矩阵是否奇异，通过计算其行列式。

首先定义一个非奇异矩阵。

```python
import numpy as np

# 一个非奇异矩阵
A_non_singular = np.array([
    [3, 1],
    [1, 2]
])

det_A = np.linalg.det(A_non_singular)
print(f"矩阵 A:\n{A_non_singular}")
print(f"A 的行列式: {det_A:.2f}")
```

**Output:**

```
矩阵 A:
[[3 1]
 [1 2]]
A 的行列式: 5.00
```

由于行列式为 5.0（不为零），这个矩阵是非奇异且可逆的。

现在我们尝试一个奇异矩阵。

```python
# 一个奇异矩阵
B_singular = np.array([
    [2, 4],
    [1, 2]
])

det_B = np.linalg.det(B_singular)
print(f"矩阵 B:\n{B_singular}")
print(f"B 的行列式: {det_B}")
```

**Output:**

```
矩阵 B:
[[2 4]
 [1 2]]
B 的行列式: 0.0
```

行列式为 0.0，确认这个矩阵是奇异的。

#### 关于浮点精度的一个注意事项

在使用计算机时，计算结果可能并非完全精确。一个理论上奇异的矩阵，其行列式可能被计算为一个非常小的数值，例如 `1.4e-17`，而非恰好 `0`。因此，在实际代码中，你不应检验行列式是否恰好等于零。而应检验它是否非常接近零。

```python
# 一个由于浮点表示而接近奇异的矩阵
C = np.array([
    [1, 2],
    [1, 2.000000000000001] 
])

det_C = np.linalg.det(C)
print(f"C 的行列式: {det_C}")

# 检验行列式是否接近零
is_singular = np.isclose(det_C, 0)
print(f"矩阵 C 是否实际为奇异？ {is_singular}")
```

**Output:**

```
C 的行列式: 8.881784197001252e-16
矩阵 C 是否实际为奇异？ True
```

使用 `np.isclose()` 是一种更可靠的方法，用于数值计算中奇异性的检验。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本经典的教材，系统地介绍了线性代数，包括行列式、矩阵可逆性及其与线性方程组解的关系。
- [Deep Learning (Chapter 2: Linear Algebra)](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  本章详细介绍了与机器学习相关的线性代数基本概念，例如矩阵性质、行列式及其对求解线性系统的影响。
- [numpy.linalg.det](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html) — NumPy Developers (2023)
  NumPy 计算矩阵行列式函数的官方文档，直接支持了文中提供的 Python 代码示例。
- [numpy.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) — NumPy Developers (2023)
  NumPy `isclose` 函数的官方文档，解释了如何在容差范围内比较浮点数，此方法对于数值稳定的奇异性检查至关重要。
