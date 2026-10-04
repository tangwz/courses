---
course: "linear-algebra-essentials-ml"
chapter: "eigenvalues-eigenvectors"
lesson: "practice-eigen-decomposition-calculations"
sourceId: 1521
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors/practice-eigen-decomposition-calculations"
title: "动手实践：特征分解计算"
description: "使用 NumPy 对示例矩阵进行特征分解并解释结果。"
order: 8
plots: ["plots/1521-0.json"]
sourceHash: "c8562b57b8776571fc282e250103d9f5800faaabbb7787ca9e284eccf8922026"
sourceCorrections: []
---

特征值 ($\lambda$) 和特征向量 (vector) ($v$) 由关系 $Av = \lambda v$ 定义。特征分解将矩阵 $A$ 表示为 $A = PDP^{-1}$，其中 $P$ 的列是特征向量，$D$ 是由特征值构成的对角矩阵。可以借助 Python 的 NumPy 库高效地进行这些计算。这是应用主成分分析（PCA）等方法的基本能力。

### 准备工作

首先，请确保您已安装 NumPy 并将其导入：

```python
import numpy as np
```

### 定义一个矩阵

让我们使用一个简单的 2x2 对称矩阵。对称矩阵具有很好的性质，例如总是拥有实数特征值并且可以对角化。

```python
# 定义一个对称矩阵
A = np.array([[4, 2],
              [2, 1]])

print("我们的矩阵 A:")
print(A)
```

### 使用 NumPy 计算特征值和特征向量 (vector)

NumPy 的线性代数模块 `linalg` 专门提供了 `eig` 函数，用于计算特征值和特征向量。

```python
# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\n特征值:")
print(eigenvalues)

print("\n特征向量（每列是一个特征向量）:")
print(eigenvectors)
```

**输出说明：**

- `eigenvalues`：这是一个一维 NumPy 数组，包含矩阵 `A` 的特征值 ($\lambda$)。在此例中，我们得到 `[5. 0.]`。
- `eigenvectors`：这是一个二维 NumPy 数组，其中每*列*代表一个特征向量，它与 `eigenvalues` 数组中相同索引位置的特征值相对应。
  - 第一列 `eigenvectors[:, 0]` 对应于 `eigenvalues[0]`。
  - 第二列 `eigenvectors[:, 1]` 对应于 `eigenvalues[1]`。

NumPy 通常返回归一化 (normalization)的特征向量（长度为 1 的向量）。

### 验证特征值方程 ($Av = \lambda v$)

让我们验证第一个特征值-特征向量 (vector)对的基本关系 $Av = \lambda v$。

```python
# 提取第一个特征值和对应的特征向量
lambda_1 = eigenvalues[0]
v_1 = eigenvectors[:, 0] # 第一列

# 计算 A * v_1
Av1 = A @ v_1 # 使用 @ 运算符进行矩阵乘法

# 计算 lambda_1 * v_1
lambda1_v1 = lambda_1 * v_1

print("\n验证第一个特征值/特征向量:")
print(f"lambda_1: {lambda_1:.4f}")
print(f"v_1: {v_1}")
print(f"A @ v_1: {Av1}")
print(f"lambda_1 * v_1: {lambda1_v1}")

# 检查 Av1 和 lambda1_v1 是否接近（由于浮点运算）
print(f"Av1 和 lambda1*v1 在数值上是否接近？ {np.allclose(Av1, lambda1_v1)}")
```

你应该会看到 `A @ v_1` 和 `lambda_1 * v_1` 的结果确实非常接近，这证实了它们之间的关系。你可以对第二个特征值和特征向量进行类似的检查。注意，当 $\lambda = 0$ 时，$Av$ 会得到零向量，这与预期相符。

### 使用特征分解重构矩阵 ($A = PDP^{-1}$)

现在，让我们使用矩阵 $A$ 的特征值和特征向量 (vector)来重构它。回想公式 $A = PDP^{-1}$，其中：

- $P$ 是以特征向量为列的矩阵。
- $D$ 是对角线上包含特征值的对角矩阵。
- $P^{-1}$ 是矩阵 $P$ 的逆矩阵。

```python
# 从特征向量构建矩阵 P
P = eigenvectors

# 从特征值构建对角矩阵 D
D = np.diag(eigenvalues)

# 计算 P 的逆
# 对于正交矩阵（如对称矩阵的特征向量），P_inv = P.T
# 但我们在这里使用 np.linalg.inv 以适应一般情况。
try:
    P_inv = np.linalg.inv(P)

    # 重构原始矩阵 A
    A_reconstructed = P @ D @ P_inv

    print("\n使用 P D P_inv 重构 A:")
    print("矩阵 P（特征向量）:")
    print(P)
    print("\n矩阵 D（对角特征值）:")
    print(D)
    print("\n矩阵 P_inv（P 的逆）:")
    print(P_inv)
    print("\n重构的矩阵 (P @ D @ P_inv):")
    print(A_reconstructed)

    # 验证重构
    print(f"\n重构的矩阵与原始矩阵 A 是否接近？ {np.allclose(A, A_reconstructed)}")

except np.linalg.LinAlgError:
    print("\n矩阵 P 是奇异的，无法求逆。A 可能无法使用此方法对角化。")
```

输出应显示重构的矩阵在数值上与原始矩阵 `A` 非常接近。这验证了特征分解。

### 变换的可视化

特征向量 (vector)表示在应用矩阵 $A$ 所代表的变换时，方向保持不变（仅发生缩放）的向量。让我们将此过程对矩阵 `A` 进行可视化。我们将变换一个标准基向量 $e_1 = [1, 0]$，并将其变换结果与第一个特征向量 $v_1$ 的变换结果进行比较。

```python
# 定义标准基向量 e1 和第一个特征向量 v1
e1 = np.array([1, 0])
# v1 已在前面的步骤中定义

# 应用变换 A
Ae1 = A @ e1
Av1 = A @ v1 # 这应该是 lambda_1 * v1

# 准备绘图数据
origin = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) # 箭头的起点
vectors = np.array([e1, Ae1, v_1, Av1]) # 向量
```



![变换 A 对向量的影响](plots/1521-0.json)



> 该图显示了原始向量（实线）以及它们经矩阵 A 变换后的结果（虚线）。蓝色向量 $e_1$ 发生了旋转和缩放。红色向量 $v_1$（一个特征向量）仅沿其原始方向按等于其特征值 ($\lambda_1 \approx 5$) 的因子进行缩放。它的变换版本 $A v_1$ 位于同一条线上。

### 重要考量

- **数值稳定性：** 直接计算 $P^{-1}$ 有时在数值上不稳定，特别是对于大型矩阵或接近奇异（不可逆）的矩阵。奇异值分解（SVD）（将在下一章中介绍）等方法在实践中通常更受青睐，用于完成相关工作。
- **不可对角化的矩阵：** 并非所有方阵都可以分解为 $PDP^{-1}$ 的形式。当矩阵不具备一组完整的线性无关特征向量 (vector)时，就会出现这种情况。NumPy 的 `np.linalg.eig` 仍可能计算特征值和向量，但此时 $P$ 将是奇异的（不可逆）。然而，对称矩阵总是可以对角化的。
- **复数特征值/特征向量：** 如果矩阵 `A` 不是对称的，它可能具有复数特征值和特征向量。`np.linalg.eig` 可以正确处理这种情况，必要时会返回复数数据类型的数组。

本次实践练习显示了 NumPy 如何简化特征值、特征向量的计算以及特征分解的验证。理解这些计算对于领会主成分分析（PCA）等算法如何借助这些特殊向量和标量来认识数据本身的结构，具有重要作用。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  全面介绍了特征值、特征向量和矩阵对角化，阐明了它们的理论和几何意义。
- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — NumPy Developers (2023)
  `numpy.linalg.eig` 函数计算特征值和特征向量的官方文档，详细说明了其用法和返回值。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  详细讨论了主成分分析（PCA），解释了其基于特征分解的公式及其在数据分析中的作用。
- [Numerical Linear Algebra](https://doi.org/10.1137/1.9780898719574) — Lloyd N. Trefethen and David Bau III (1997)
  Publisher: SIAM; Pages: xii+361; DOI: [10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
  涵盖了特征值问题的数值算法，包括对稳定性、效率和高级分解方法的讨论。
