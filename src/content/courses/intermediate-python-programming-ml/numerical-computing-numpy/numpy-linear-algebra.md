---
course: "intermediate-python-programming-ml"
chapter: "numerical-computing-numpy"
lesson: "numpy-linear-algebra"
sourceId: 2126
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/numpy-linear-algebra"
title: "NumPy 中的线性代数运算"
description: "使用 NumPy 执行矩阵乘法和分解等基本线性代数运算。"
order: 6
plots: []
sourceHash: "c30c6ee86d09e85c334ed177e22d60918993e484cbef6b01ddb46cfa3cd08d02"
sourceCorrections: []
---

线性代数是许多机器学习 (machine learning)算法的根本。从表示数据集和模型参数 (parameter)，到执行变换和解决优化问题，向量 (vector)和矩阵的运算都是不可或缺的。NumPy 通过其 `linalg` 模块，提供了一套全面且高度优化的函数，用于完成这些必要的线性代数任务。此处描述的方法使用 NumPy 数组执行这些计算。

请记住，一维 NumPy 数组可以表示向量，二维数组可以表示矩阵。

```python
import numpy as np

# 向量 (一维数组)
v = np.array([1, 2, 3])
print("向量 v:\n", v)

# 矩阵 (二维数组)
M = np.array([[1, 2], [3, 4], [5, 6]])
print("\n矩阵 M:\n", M)
print("\nM 的形状:", M.shape) # (3, 2) -> 3 行, 2 列
```

### 矩阵乘法

矩阵乘法是机器学习 (machine learning)中经常执行的运算之一。它应用于神经网络 (neural network)中的权重 (weight)计算，以及特征空间变换等所有方面。区分元素级乘法和真正的矩阵乘法（点积）很关键。

- **元素级乘法：** 使用 `*` 运算符。要求数组的形状根据广播规则兼容。第一个数组中的每个元素都与第二个数组中的对应元素相乘。
- **矩阵乘法（点积）：** 使用 `@` 运算符（Python 3.5+ 中更常用）或 `np.dot()` 函数。对于两个矩阵 $A$ 和 $B$，只有当 $A$ 的列数等于 $B$ 的行数时，乘积 $AB$ 才成立。如果 $A$ 是一个 $m \times n$ 矩阵，$B$ 是一个 $n \times p$ 矩阵，则它们的乘积 $C = AB$ 将是一个 $m \times p$ 矩阵。

```python
A = np.array([[1, 2], [3, 4]]) # 2x2 矩阵
B = np.array([[5, 6], [7, 8]]) # 2x2 矩阵
v = np.array([9, 10])          # 向量 (根据上下文，可视为 1x2 或 2x1)

# 元素级乘法
print("元素级 A * B:\n", A * B)

# 矩阵乘法
print("\n矩阵乘法 A @ B:\n", A @ B)
print("\n使用 np.dot(A, B) 进行矩阵乘法:\n", np.dot(A, B))

# 矩阵-向量乘法
# NumPy 在此情况下自动将 v 视为列向量
print("\n矩阵-向量乘法 A @ v:\n", A @ v) # 结果是一维数组
```

矩阵乘法中兼容形状的规则（$m \times n$ 乘以 $n \times p$ 得到 $m \times p$）非常重要。

> 此图表示矩阵乘法维度兼容性。

### 矩阵的转置

矩阵的转置运算会交换其行和列。如果 $A$ 是一个 $m \times n$ 矩阵，它的转置（表示为 $A^T$）是一个 $n \times m$ 矩阵，其中 $(A^T)_{ij} = A_{ji}$。在 NumPy 中，您可以使用 `.T` 属性或 `np.transpose()` 函数获取转置。

```python
M = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 矩阵
print("原始矩阵 M:\n", M)
print("\nM 的形状:", M.shape)

# 使用 .T 属性转置
M_transpose = M.T
print("\n转置 M.T:\n", M_transpose)
print("\nM.T 的形状:", M_transpose.shape) # 3x2 矩阵

# 使用 np.transpose() 函数转置
M_transpose_func = np.transpose(M)
print("\n转置 np.transpose(M):\n", M_transpose_func)
print("\nnp.transpose(M) 的形状:", M_transpose_func.shape) # 3x2 矩阵
```

转置常用于处理方程，或根据形状规则对齐 (alignment)向量 (vector)和矩阵以进行乘法运算。

### 矩阵的逆和行列式

方阵 $A$ 的逆（表示为 $A^{-1}$）是一个矩阵，当它与 $A$ 相乘时，结果是单位矩阵 $I$ ($AA^{-1} = A^{-1}A = I$)。矩阵必须是方阵（行数和列数相同）且非奇异（其行列式非零）才能有逆矩阵。逆矩阵对于求解线性方程组非常重要。

行列式是一个标量值，可以从方阵的元素计算得出，并提供有关该矩阵的重要信息，例如它是否可逆。

NumPy 的 `np.linalg` 模块提供以下函数：

- `np.linalg.inv(A)`：计算矩阵 A 的逆。
- `np.linalg.det(A)`：计算矩阵 A 的行列式。

```python
# 创建一个可逆方阵
A = np.array([[1, 2], [3, 4]])
print("矩阵 A:\n", A)

# 计算行列式
det_A = np.linalg.det(A)
print("\nA 的行列式:", det_A) # 应为 1*4 - 2*3 = -2

# 计算逆
inv_A = np.linalg.inv(A)
print("\nA 的逆:\n", inv_A)

# 验证 A @ A_inv 是否接近单位矩阵
identity = np.eye(2) # 2x2 单位矩阵
print("\nA @ inv_A (应接近单位矩阵):\n", A @ inv_A)
# 注意：由于浮点精度问题，结果可能非常接近但并非完全是单位矩阵。
print("\nA @ inv_A 是否接近单位矩阵?", np.allclose(A @ inv_A, identity))
```

如果您尝试计算奇异矩阵（行列式为 0）的逆，NumPy 将引发 `LinAlgError`。

```python
# 奇异矩阵（第2列是第1列的2倍）
singular_M = np.array([[1, 2], [2, 4]])
print("\n奇异矩阵:\n", singular_M)
print("行列式:", np.linalg.det(singular_M)) # 应为 0 或因浮点精度而非常接近 0

try:
    inv_singular = np.linalg.inv(singular_M)
    print("逆（不应打印此行）:\n", inv_singular)
except np.linalg.LinAlgError as e:
    print("\n计算逆时出错:", e)
```

在实践中，特别是在涉及可能非方阵或奇异矩阵的机器学习 (machine learning)情境中（例如具有冗余特征的线性回归），伪逆 (`np.linalg.pinv`) 通常作为逆矩阵的通用形式使用。

### 求解线性方程组

在包括机器学习 (machine learning)在内的各种科学方面中，一个常见的问题是求解线性方程组。这样的系统可以表示为矩阵形式：

$Ax = b$

其中 $A$ 是一个已知的系数方阵，$x$ 是我们希望求出的未知数向量 (vector)，$b$ 是一个已知的列向量。

如果 $A$ 是可逆的，找到 $x$ 的一种方法是两边乘以 $A$ 的逆矩阵：

$A^{-1}Ax = A^{-1}b$
$Ix = A^{-1}b$
$x = A^{-1}b$

虽然您可以使用 `np.linalg.inv(A) @ b` 进行计算，但通常**不**推荐这样做。计算逆矩阵的计算成本更高，而且数值稳定性可能不如专用求解器。NumPy 提供了 `np.linalg.solve(A, b)`，它使用更高效、更稳定的算法（通常基于 LU 分解）直接求得 $x$。

考虑以下系统：
$x_1 + 2x_2 = 1$
$3x_1 + 4x_2 = -1$

以矩阵形式表示：


$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
=
\begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$


我们来使用 NumPy 求解：

```python
# 系数矩阵 A
A = np.array([[1, 2], [3, 4]])

# 常数向量 b
b = np.array([1, -1])

print("矩阵 A:\n", A)
print("\n向量 b:", b)

# 使用 np.linalg.solve 求解
x = np.linalg.solve(A, b)
print("\n使用 solve() 的解向量 x:", x) # 预期结果: [-3, 2]

# 验证解：A @ x 应该接近 b
print("\n验证 A @ x:", A @ x)
print("A @ x 是否接近 b?", np.allclose(A @ x, b))

# 作比较：使用显式逆矩阵求解（不太推荐）
inv_A = np.linalg.inv(A)
x_inv = inv_A @ b
print("\n使用逆矩阵的解向量 x:", x_inv)
```

对于条件良好的方阵，两种方法都能给出相同的结果，但 `np.linalg.solve` 是标准且推荐的方法。

### 其他有用的线性代数函数

`np.linalg` 模块包含许多其他函数。以下是几个与机器学习 (machine learning)特别相关的函数：

- **特征值和特征向量 (vector) (`np.linalg.eig(A)`)：** 对于方阵 $A$，一个特征向量 $v$ 和对应的特征值 $\lambda$ 满足方程 $Av = \lambda v$。特征值和特征向量对于理解矩阵变换非常重要，并广泛应用于诸如主成分分析（PCA）等降维算法中，其中特征向量指示最大方差的方向，特征值指示这些方向上变化的大小。
- **范数 (`np.linalg.norm(x, ord=...)`)：** 范数是衡量向量或矩阵“大小”或“长度”的量度。存在不同类型的范数（由 `ord` 参数 (parameter)指定）。常见的向量范数包括：
  - L2 范数（欧几里得范数，默认）：`ord=2` 或 `ord=None`。计算 $\sqrt{\sum_i x_i^2}$。常用于衡量距离或误差。
  - L1 范数（曼哈顿范数）：`ord=1`。计算 $\sum_i |x_i|$。用于正则化 (regularization)（Lasso）以促使稀疏性。
    矩阵范数（如 Frobenius 范数 (`ord='fro'`)）也可用。范数在 Ridge 和 Lasso 回归等模型的正则化技术、评估模型误差以及 k-近邻算法中的距离计算方面很重要。

```python
# 示例：计算范数
v = np.array([3, -4])
print("\n向量 v:", v)

# L2 范数（默认）
norm_l2 = np.linalg.norm(v)
print("L2 范数（欧几里得）:", norm_l2) # sqrt(3^2 + (-4)^2) = sqrt(9+16) = sqrt(25) = 5.0

# L1 范数
norm_l1 = np.linalg.norm(v, ord=1)
print("L1 范数（曼哈顿）:", norm_l1) # |3| + |-4| = 3 + 4 = 7.0

# 示例：特征值和特征向量
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n矩阵 A:\n", A)
print("\n特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)

# 验证第一个特征值/向量对：A @ v = lambda * v
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0] # 第一列是第一个特征向量
print("\nA @ v1:", A @ v1)
print("lambda1 * v1:", lambda1 * v1)
print("它们接近吗?", np.allclose(A @ v1, lambda1 * v1))
```

掌握这些 NumPy 线性代数运算对于实现和理解许多机器学习算法很重要。它们提供了高效处理数据表示和模型参数的计算组成部分。请记住，NumPy 的实现是高度优化的，与手动 Python 循环相比，为这些计算提供了显著的速度优势。

## 参考资料

- [NumPy Reference - Linear algebra (numpy.linalg)](https://numpy.org/doc/stable/reference/routines.linalg.html) — NumPy Developers (2023)
  NumPy线性代数模块的官方文档，提供所讨论所有函数的详细说明和使用示例。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebras/) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press; Pages: 440
  一本基础教科书，全面介绍了线性代数的数学原理，涵盖向量、矩阵、行列式、逆矩阵和特征值等主题。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  涵盖专为机器学习设计的核心线性代数概念，包括矩阵运算、分解及其实际应用。
- [18.06SC Linear Algebra, Fall 2011](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  一个免费在线课程，提供完整的本科线性代数课程的视频讲座、讲义和习题，与常见的机器学习先决条件相符。
