---
course: "linear-algebra-essentials-ml"
chapter: "matrix-decompositions-machine-learning"
lesson: "implementing-decompositions-scipy-numpy"
sourceId: 1541
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/implementing-decompositions-scipy-numpy"
title: "使用 SciPy/NumPy 实现分解"
description: "使用 SciPy 和 NumPy 库进行 SVD、LU 和 QR 分解。"
order: 9
plots: []
sourceHash: "38517b39a9e7aa06ad4bb38d8043f23421843c6c66cf0ccefcbff3f747811d48"
sourceCorrections: []
---

SVD、LU 和 QR 等矩阵分解背后的理论很重要，但手动进行这些计算对于任何大尺寸的矩阵都不切实际。幸运的是，Python 的科学计算库，特别是 NumPy 和 SciPy，提供了这些算法的高效且数值稳定的实现。

我们将主要使用 `scipy.linalg` 中的函数，该模块通常比 `numpy.linalg` 提供更多高级功能和优化，尽管 NumPy 也提供了基础版本。请确保您已安装 NumPy 和 SciPy 并导入它们：

```python
import numpy as np
from scipy import linalg
```

### 奇异值分解 (SVD)

SVD 将矩阵 $A$ 分解为 $U \Sigma V^T$。`scipy.linalg.svd` 函数处理此分解。

```python
# 定义一个示例矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("原始矩阵 A ({}):".format(A.shape))
print(A)

# 执行 SVD
# full_matrices=False 计算精简 SVD（对于非方阵更高效）
U, s, Vh = linalg.svd(A, full_matrices=False)

print("\nU 矩阵 ({}):".format(U.shape))
print(U)

# s 包含奇异值（一维数组）
print("\n奇异值 (s):", s.shape)
print(s)

# Vh 是 V 的转置（V.T）
print("\nVh 矩阵 (V 转置) ({}):".format(Vh.shape))
print(Vh)
```

**理解输出：**

- `U`：左奇异向量 (vector)矩阵。其形状取决于 `full_matrices`。对于 $m \times n$ 矩阵且 $m \ge n$，当 `full_matrices=False` 时，$U$ 的形状将是 $m \times n$。
- `s`：一个包含按降序排列的奇异值（$\sigma_1, \sigma_2, ..., \sigma_k$）的一维 NumPy 数组。它*不是*对角矩阵 $\Sigma$。
- `Vh`：右奇异向量矩阵，已转置（$V^T$）。其形状取决于 `full_matrices`。对于 $m \ge n$，当 `full_matrices=False` 时，$Vh$ 的形状将是 $n \times n$。

**重构原始矩阵（验证）：**

为了验证分解 $A = U \Sigma V^T$，您需要从奇异值 `s` 构建对角矩阵 $\Sigma$。

```python
# 构建 Sigma 矩阵（对角矩阵）
# 需要匹配维度以进行乘法
Sigma = np.zeros(A.shape) # 以 A 的形状创建零矩阵
# 用奇异值填充对角线
# min(A.shape) 给出奇异值的数量
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s) # 此处使用 Vh 的形状 (n x n)

# 重构 A
A_reconstructed = U @ Sigma @ Vh # 使用矩阵乘法运算符 @

print("\nSigma 矩阵 ({}):".format(Sigma.shape))
print(Sigma)

print("\n重构矩阵 (U @ Sigma @ Vh):")
print(A_reconstructed)

# 检查是否与原始矩阵接近（由于浮点精度）
print("\n重构矩阵是否接近原始矩阵？", np.allclose(A, A_reconstructed))
```

`np.allclose()` 函数适用于比较浮点数组，同时考虑小的数值误差。当 $m \neq n$ 时，通常建议使用 `full_matrices=False`（精简 SVD），因为它避免了在 $U$ 或 $V$ 中计算不必要的零向量。

### LU 分解

LU 分解将一个方阵 $A$ 分解为 $L$（下三角矩阵）和 $U$（上三角矩阵），使得 $A = LU$。通常，为了稳定性需要进行主元选择（行交换），从而得到 $PA = LU$，其中 $P$ 是一个置换矩阵。`scipy.linalg.lu` 处理此过程。

```python
# 定义一个方阵 B
B = np.array([[2, 5, 8],
              [4, 2, 1],
              [9, 3, 7]])

print("原始矩阵 B ({}):".format(B.shape))
print(B)

# 执行带置换的 LU 分解
P_mat, L, U = linalg.lu(B)

# 注意：P_mat 就是置换矩阵本身
print("\n置换矩阵 P ({}):".format(P_mat.shape))
print(P_mat)

print("\n下三角矩阵 L ({}):".format(L.shape))
print(L)

print("\n上三角矩阵 U ({}):".format(U.shape))
print(U)

# 验证：P @ B 应接近 L @ U
print("\nP @ B 是否接近 L @ U？", np.allclose(P_mat @ B, L @ U))
```

**理解输出：**

- `P_mat`：置换矩阵 $P$。将其应用于 $A$（作为 $P @ A$）表示高斯消元过程中执行的行交换。
- `L`：对角线上为一的下三角矩阵。
- `U`：上三角矩阵。

LU 分解是线性方程组（$Ax=b$）高效直接求解器的一个重要方法，尽管像 `scipy.linalg.solve` 这样的函数通常会将其抽象化。

### QR 分解

QR 分解将任意 $m \times n$ 矩阵 $A$ 分解为 $Q$（一个正交矩阵，$Q^T Q = I$）和 $R$（一个上三角矩阵），使得 $A = QR$。`scipy.linalg.qr` 是应使用的函数。

```python
# 再次使用原始矩阵 A
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9],
#               [10, 11, 12]])
print("原始矩阵 A ({}):".format(A.shape))
print(A)

# 执行 QR 分解
# mode='economic' 对于非方阵更高效
Q, R = linalg.qr(A, mode='economic')

print("\n正交矩阵 Q ({}):".format(Q.shape))
print(Q)

print("\n上三角矩阵 R ({}):".format(R.shape))
print(R)

# 验证 Q 的正交性：Q.T @ Q 应接近单位矩阵
I = np.eye(Q.shape[1]) # 适当大小的单位矩阵
print("\nQ.T @ Q 是否接近单位矩阵？", np.allclose(Q.T @ Q, I))

# 验证分解：A 应接近 Q @ R
print("\nA 是否接近 Q @ R？", np.allclose(A, Q @ R))
```

**理解输出：**

- `Q`：正交（对于复数矩阵是酉）矩阵。当 `mode='economic'` 时，如果 $m \ge n$，则 $Q$ 是 $m \times n$；如果 $m < n$，则 $Q$ 是 $m \times m$。其列构成 $A$ 的列空间的正交规范基。
- `R`：上三角矩阵。当 `mode='economic'` 时，如果 $m \ge n$，则 $R$ 是 $n \times n$；如果 $m < n$，则 $R$ 是 $m \times n$。

QR 分解在用于解决最小二乘问题、特征值计算（QR 算法）和正交化过程的算法中具有重要作用。这些 SciPy 函数提供了应用矩阵分解的工具。它们处理了数值复杂性，使您能够专注于这些分解如何用于解决机器学习 (machine learning)中的问题，例如降维（SVD）、求解线性系统（LU）或寻找最小二乘解（QR）。下一节将提供一个侧重于 SVD 的实际示例。

## 参考资料

- [SciPy Reference Guide: scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html) — SciPy Developers (2023)
  这份官方文档提供了 `scipy.linalg` 模块的详细信息，包括内容中使用的 SVD、LU 和 QR 分解函数及其参数和返回值。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  这本经典教材从基础原理出发，对包括 SVD、LU 和 QR 在内的矩阵分解提供了全面的理论解释。
- [Numerical Linear Algebra](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHum5eGqcWZ6PpxuoG-1jlQSGtvl8-yXvmq1PoKSK_7uWSB_ZfS0xEyivdwAj-szHZALYG7n3wyw_JnfzkrnNJJ5I3zLG8SBImdI2aVhNgIY7SeYQO8px5Vsxnb3-QtiqOmoonULQopD7AvFww=) — Lloyd N. Trefethen and David Bau III (1997)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9781611971212](https://doi.org/10.1137/1.9781611971212)
  这本标准参考文献有助于理解矩阵分解背后的数值方法和稳定性，这对于高效的软件实现至关重要。
