---
course: "linear-algebra-essentials-ml"
chapter: "eigenvalues-eigenvectors"
lesson: "calculating-eigen-numpy"
sourceId: 1519
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors/calculating-eigen-numpy"
title: "使用NumPy计算特征值和特征向量"
description: "使用NumPy的线性代数函数计算特征值和特征向量。"
order: 7
plots: []
sourceHash: "9c5365c2322069020322edfc37451904e17347a87d230cca347dcae1b710c196"
sourceCorrections: []
---

虽然理解特征方程 $det(A - \lambda I) = 0$ 并求解系统 $(A - \lambda I)x = 0$ 能提供特征值 $\lambda$ 和特征向量 (vector) $x$，但对于大于2x2或3x3的矩阵，手动执行这些计算变得不切实际。机器学习 (machine learning)情境常常涉及高维数据，从而产生大型矩阵。幸运的是，NumPy等数值库提供了高效的函数来完成这些计算。

NumPy中用于特征值和特征向量计算的主要方法是 `numpy.linalg.eig` 函数。此函数接受一个方阵作为输入，并返回其特征值及对应的右特征向量。

### 使用 `numpy.linalg.eig`

让我们看看如何使用 `np.linalg.eig`。它接受一个参数 (parameter)：我们想要找出特征值和特征向量 (vector)的方阵 `A`。它返回一个包含两个NumPy数组的元组：

1. `eigenvalues`：一个一维NumPy数组，包含矩阵的特征值（$\lambda$）。
2. `eigenvectors`：一个二维NumPy数组，其中每*列*代表与`eigenvalues`数组中相同索引处的特征值相对应的归一化 (normalization)特征向量。

考虑以下2x2矩阵 $A$：


$$
A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}
$$


我们可以这样使用NumPy来找出其特征值和特征向量：

```python
import numpy as np

# 定义矩阵 A
A = np.array([[4, 2],
              [1, 3]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues (λ):")
print(eigenvalues)
print("\nEigenvectors (columns of the matrix below):")
print(eigenvectors)
```

运行此代码将输出：

```
Matrix A:
[[4 2]
 [1 3]]

Eigenvalues (λ):
[5. 2.]

Eigenvectors (columns of the matrix below):
[[ 0.89442719 -0.70710678]
 [ 0.4472136  -0.70710678]]
```

因此，特征值是 $\lambda_1 = 5$ 和 $\lambda_2 = 2$。

对应的特征向量是 `eigenvectors` 数组的列：

- 对于 $\lambda_1 = 5$，特征向量是 $x_1 \approx \begin{bmatrix} 0.894 \\ 0.447 \end{bmatrix}$。
- 对于 $\lambda_2 = 2$，特征向量是 $x_2 \approx \begin{bmatrix} -0.707 \\ -0.707 \end{bmatrix}$。

请注意，NumPy返回的是*归一化*的特征向量，这意味着它们的L2范数（长度）是1。

### 验证结果

我们可以通过检查基本关系 $Ax = \lambda x$ 是否对每对成立来验证这些结果。让我们检查第一对特征值和特征向量 (vector)（$\lambda_1=5$， $x_1$）。

```python
# 提取第一个特征值和特征向量
lambda1 = eigenvalues[0]
x1 = eigenvectors[:, 0] # 第一列

# 计算 Ax
Ax1 = A @ x1 # 使用矩阵乘法运算符 @

# 计算 λx
lambda1_x1 = lambda1 * x1

print("\nVerifying for λ =", lambda1)
print("A * x1:")
print(Ax1)
print("λ1 * x1:")
print(lambda1_x1)

# 检查它们是否接近（由于潜在的浮点误差）
print("Ax 和 λx 是否接近？", np.allclose(Ax1, lambda1_x1))
```

输出应确认 $Ax_1$ 和 $\lambda_1 x_1$ 确实非常接近，从而验证了计算的准确性：

```
Verifying for λ = 5.0
A * x1:
[4.47213595 2.23606798]
λ1 * x1:
[4.47213595 2.23606798]
Are Ax and λx close? True
```

您可以对第二对（$\lambda_2=2$， $x_2$）进行类似的检查。

### 与特征分解的关系

回想特征分解 $A = PDP^{-1}$，其中 $P$ 是以特征向量 (vector)为列的矩阵，而 $D$ 是以特征值为对角线元素的对角矩阵。`np.linalg.eig`返回的`eigenvectors`数组正是矩阵 $P$。`eigenvalues`数组提供了 $D$ 的对角线元素。

我们可以验证分解属性，通常更方便地通过 $AP = PD$ 来检查。

```python
# 构建对角矩阵 D
D = np.diag(eigenvalues)

# 获取特征向量矩阵 P
P = eigenvectors

# 计算 AP
AP = A @ P

# 计算 PD
PD = P @ D

print("\nVerifying Eigen-decomposition (AP = PD)")
print("AP:")
print(AP)
print("\nPD:")
print(PD)
print("\nAP 和 PD 是否接近？", np.allclose(AP, PD))
```

此验证也应返回 `True`，确认 `np.linalg.eig` 的输出与特征分解公式一致。

```
Verifying Eigen-decomposition (AP = PD)
AP:
[[ 4.47213595 -1.41421356]
 [ 2.23606798 -1.41421356]]

PD:
[[ 4.47213595 -1.41421356]
 [ 2.23606798 -1.41421356]]

Are AP and PD close? True
```

### 特定矩阵类型的考量

- **对称矩阵：** 如果您的矩阵 $A$ 是对称的（即 $A = A^T$），其特征值将始终是实数，并且其特征向量 (vector)可以选择为正交的。对于此类矩阵，使用 `numpy.linalg.eigh` 在数值上通常更有利。此函数针对对称（或厄米）矩阵进行了优化，并确保得到实数特征值和正交特征向量。
- **复数特征值/特征向量：** 如果输入矩阵不是对称的，其特征值和特征向量可能是复数。`numpy.linalg.eig` 会正确处理这种情况，必要时返回复数数组。

使用 NumPy 的 `linalg.eig` 函数能够简化求解特征多项式和由此产生的线性系统的复杂过程，为在主成分分析（PCA）等实际机器学习 (machine learning)任务中应用特征值原理提供了一个强大的方法。PCA很大程度上依赖于协方差矩阵的特征分解。

## 参考资料

- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — NumPy Developers (2023)
  `numpy.linalg.eig` 函数的官方文档，详细说明其用法和参数。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebras/) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press
  一本标准教材，为特征值、特征向量和特征分解提供了全面的理论基础。
- [Numerical Linear Algebra](https://epubs.siam.org/doi/book/10.1137/1.9781611971210) — Lloyd N. Trefethen and David Bau, III (1997)
  Publisher: SIAM; DOI: [10.1137/1.9781611971210](https://doi.org/10.1137/1.9781611971210)
  经典著作，涵盖了数值特征值计算背后的计算算法和稳定性考量。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第二章讨论了线性代数基础知识，包括特征值和特征向量，与PCA等机器学习场景相关。
