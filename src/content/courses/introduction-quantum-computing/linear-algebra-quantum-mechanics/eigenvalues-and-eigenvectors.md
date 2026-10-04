---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "eigenvalues-and-eigenvectors"
sourceId: 8150
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/eigenvalues-and-eigenvectors"
title: "特征值与特征向量"
description: "测量结果与特征值之间的关系。"
order: 5
plots: []
sourceHash: "37d9cd12a52bb305411bb2f2145b7ea1e1038f41111cbe85f09960cc99dbb441"
sourceCorrections: []
---

矩阵充当变换向量 (vector)的函数。当我们将一个向量乘以一个矩阵时，得到的向量通常会改变其大小（长度）和方向。然而，对于每个方阵，都存在一些表现特殊的向量。当矩阵作用于这些向量时，其方向保持不变。向量可能会拉伸或收缩，但它仍保持在原来的跨度上。

这些特殊的向量被称为**特征向量 (eigenvectors)**。它们拉伸或收缩的比例被称为**特征值 (eigenvalue)**。

这一原理是量子测量背后的数学引擎。在前一节中，我们讨论了作为操作的酉矩阵。当我们测量一个量子系统时，本质上是要求系统塌缩到其特征向量之一。我们从测量中读出的数值就是对应的特征值。

### 特征方程

矩阵 $A$、特征向量 (vector) $v$ 和特征值 $\lambda$ (lambda) 之间的关系由以下方程定义：


$$
Av = \lambda v
$$


这个方程告诉我们：

1. **$A$** 是变换矩阵（算符）。
2. **$v$** 是非零向量（特征向量）。
3. **$\lambda$** 是标量值（特征值）。

在等式左侧，我们进行的是矩阵与向量的乘法。在右侧，则是标量与向量的乘法。这意味着矩阵 $A$ 对向量 $v$ 的作用等同于简单地将 $v$ 按因子 $\lambda$ 进行缩放。

> 矩阵 A 对向量 v 的变换结果是一个指向相同方向的向量。

### 量子系统中的物理意义

在经典机器学习 (machine learning)中，特征向量 (vector)有助于降维（如主成分分析）。在量子计算中，它们的作用则是具体且物理化的。

量子系统中每一个可测量的属性（如能量、自旋或位置）都与一个算符（矩阵）相关联。我们称之为**可观测波 (Observables)**。当你进行测量时：

1. 系统塌缩到该算符的其中一个**特征向量**上。
2. 你在测量设备上看到的结果就是该特征向量对应的**特征值**。

这解释了为什么量子结果是量子化的。如果一个算符的特征值只有 $+1$ 和 $-1$，你永远不会测得 $0.5$ 这个值。你只会观察到 $+1$ 或 $-1$。

### 具体示例：Pauli-Z 算符

让我们来看看 Pauli-Z 矩阵，它通常用于在计算基（$0$ 和 $1$）下测量量子比特的状态。该矩阵定义为：


$$
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$


我们想看看这个矩阵如何作用于标准基向量 (vector)。首先，看状态 $|0\rangle$，由向量 $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ 表示。


$$
Z|0\rangle = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \cdot 1 + 0 \cdot 0 \\ 0 \cdot 1 + (-1) \cdot 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$


输出与输入完全相同。


$$
Z|0\rangle = 1|0\rangle
$$


向量 $|0\rangle$ 是 $Z$ 矩阵的一个特征向量，对应的特征值为 $1$。

现在考虑状态 $|1\rangle$，由 $\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ 表示。


$$
Z|1\rangle = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ -1 \end{pmatrix} = -1 \begin{pmatrix} 0 \\ 1 \end{pmatrix}
$$


在这里，向量保持在同一轴线上，但指向相反的方向。


$$
Z|1\rangle = -1|1\rangle
$$


向量 $|1\rangle$ 是 $Z$ 矩阵的一个特征向量，对应的特征值为 $-1$。

如果我们使用 $Z$ 矩阵来测量一个量子比特，唯一可能的结果是 $1$ 和 $-1$。状态 $|0\rangle$ 对应结果 $1$，状态 $|1\rangle$ 对应结果 $-1$。

### 使用 NumPy 进行计算

虽然手动求解 $2 \times 2$ 矩阵的特征方程很有用，但对于更大的系统，我们需要依靠计算工具。Python 的 NumPy 库有一个内置的线性代数模块 `linalg`，可以高效地处理这些计算。

我们可以使用 `np.linalg.eig()` 来计算矩阵的特征值和特征向量 (vector)。

```python
import numpy as np

# 定义 Pauli-Z 矩阵
Z = np.array([[1, 0], 
              [0, -1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(Z)

print("特征值:")
print(eigenvalues)

print("\n特征向量:")
print(eigenvectors)
```

运行这段代码会得到以下输出：

```text
特征值:
[ 1. -1.]

特征向量:
[[1. 0.]
 [0. 1.]]
```

`eigenvalues` 数组包含 `1` 和 `-1`。`eigenvectors` 数组将对应的向量作为列包含在内。第一列 `[1, 0]` 对应第一个特征值 `1`。第二列 `[0, 1]` 对应第二个特征值 `-1`。

### 为什么这对算法很重要

理解特征值对于理解量子算法如何提取信息是必要的。许多量子算法，例如量子相位估计（Shor 算法中用于因式分解的一个组成部分），都依赖于对系统的操作，使得问题的答案被编码在酉算符的特征值中。

当量子比特处于叠加态（基态的混合）时，它不是测量算符的特征向量 (vector)。在这种情况下，测量结果是概率性的。系统会随机跳转到其中一个特征向量，其概率取决于当前状态与这些特征向量之间的重叠程度。我们将在下一章构建包含叠加和测量的电路时验证这一行为。
