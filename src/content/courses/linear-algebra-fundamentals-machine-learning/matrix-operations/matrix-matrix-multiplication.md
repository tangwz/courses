---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "matrix-operations"
lesson: "matrix-matrix-multiplication"
sourceId: 7817
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-3-matrix-operations/matrix-matrix-multiplication"
title: "矩阵-矩阵乘法"
description: "逐步讲解两个矩阵相乘的规则和过程。"
order: 4
plots: []
sourceHash: "1c47d47f02fb1cabc1ee529c35323877553bb3d4567acdb21b8982c6d0c47193"
sourceCorrections: []
---

将矩阵与向量 (vector)相乘，可以对空间中的单个点进行变换，而将矩阵与另一个矩阵相乘，则如同将两个变换组合为一个变换。这个运算在机器学习 (machine learning)中非常基本，特别是在神经网络 (neural network)中，数据通过连续层时，其本质就是一系列的矩阵乘法。

### 乘法规则：内维度匹配

在将两个矩阵相乘之前，它们必须是兼容的。这是需要记住的主要规则。如果你有一个维度为 $m \times n$（表示 $m$ 行 $n$ 列）的矩阵 $A$ 和一个维度为 $n \times p$ 的矩阵 $B$，你可以将它们相乘，得到一个维度为 $m \times p$ 的新矩阵 $C$。

规则很简单：**第一个矩阵的列数必须等于第二个矩阵的行数。** 我们称这些为“内维度”。


$$
A_{m \times n} \cdot B_{n \times p} = C_{m \times p}
$$


“外维度” $m$ 和 $p$ 决定了最终矩阵的形状。如果内维度不匹配，则乘法未定义。

> 第一个矩阵的列数必须与第二个矩阵的行数对齐 (alignment)，乘法才能有效。结果矩阵会继承第一个矩阵的行数和第二个矩阵的列数。

### 如何计算乘积

实际计算是你之前学过的矩阵-向量 (vector)乘法的延伸。要获得结果矩阵 $C$ 中第 $i$ 行第 $j$ 列的元素，你需要计算矩阵 $A$ 的第 $i$ 行与矩阵 $B$ 的第 $j$ 列的**点积**。

我们来看一个例子。假设我们要计算乘积 $C = AB$，其中：


$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \quad \text{且} \quad B = \begin{pmatrix} 7 & 8 \\ 9 & 1 \\ 2 & 3 \end{pmatrix}
$$


首先，检查维度。$A$ 是一个 $2 \times 3$ 矩阵，$B$ 是一个 $3 \times 2$ 矩阵。内维度匹配（3 和 3），所以运算有效。结果矩阵 $C$ 的维度将是 $2 \times 2$。


$$
C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$


现在我们来计算 $C$ 的每个元素：

- **计算 $C_{11}$（第1行，第1列）：**
  取 $A$ 的第1行与 $B$ 的第1列的点积。

  
  $$
  C_{11} = (1 \cdot 7) + (2 \cdot 9) + (3 \cdot 2) = 7 + 18 + 6 = 31
  $$
  
- **计算 $C_{12}$（第1行，第2列）：**
  取 $A$ 的第1行与 $B$ 的第2列的点积。

  
  $$
  C_{12} = (1 \cdot 8) + (2 \cdot 1) + (3 \cdot 3) = 8 + 2 + 9 = 19
  $$
  
- **计算 $C_{21}$（第2行，第1列）：**
  取 $A$ 的第2行与 $B$ 的第1列的点积。

  
  $$
  C_{21} = (4 \cdot 7) + (5 \cdot 9) + (6 \cdot 2) = 28 + 45 + 12 = 85
  $$
  
- **计算 $C_{22}$（第2行，第2列）：**
  取 $A$ 的第2行与 $B$ 的第2列的点积。

  
  $$
  C_{22} = (4 \cdot 8) + (5 \cdot 1) + (6 \cdot 3) = 32 + 5 + 18 = 55
  $$
  

综上所述，我们的最终矩阵是：


$$
C = \begin{pmatrix} 31 & 19 \\ 85 & 55 \end{pmatrix}
$$


### 矩阵乘法的重要性质

矩阵乘法有一些不同于普通数字（标量）乘法的性质。

#### 顺序很重要：它不满足交换律

对于标量，$3 \times 5$ 和 $5 \times 3$ 是相同的。但这不适用于矩阵。一般来说，$AB \neq BA$。这是最明显的差异之一。

沿用我们之前的例子，我们尝试计算 $BA$：


$$
B_{3 \times 2} \cdot A_{2 \times 3}
$$


内维度（2 和 2）匹配，所以我们可以执行这个乘法。结果将是一个 $3 \times 3$ 矩阵，这与我们从 $AB$ 得到的 $2 \times 2$ 矩阵形状不同。由于结果形状不同，它们不可能相等。即使形状相同，数值也可能不同。

#### 它满足结合律

虽然矩阵的顺序不能互换，但只要矩阵本身保持相同的序列，运算的顺序就不重要。这被称为结合律：


$$
(AB)C = A(BC)
$$


这很有用，因为它表示你可以以任何计算高效的方式对矩阵乘法进行分组，而不会改变最终结果。

### NumPy 中的矩阵乘法

NumPy 使矩阵乘法变得简单。现代且推荐的方法是使用 `@` 运算符，它是专门为矩阵乘法引入的。

```python
import numpy as np

# 定义我们例子中的矩阵 A 和 B
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 1],
    [2, 3]
])

# 使用 @ 运算符执行矩阵乘法
C = A @ B

print("矩阵 A (2x3):\n", A)
print("\n矩阵 B (3x2):\n", B)
print("\nA @ B 的结果 (2x2):\n", C)
```

**输出：**

```
Matrix A (2x3):
 [[1 2 3]
 [4 5 6]]

Matrix B (3x2):
 [[7 8]
 [9 1]
 [2 3]]

Result of A @ B (2x2):
 [[31 19]
 [85 55]]
```

结果与我们手动计算的完全一致。

你也可能会看到 `np.dot()` 函数用于矩阵乘法。它既适用于向量 (vector)点积，也适用于矩阵乘法，这有时可能会造成混淆。

```python
# 使用 np.dot() 同样有效
C_dot = np.dot(A, B)
print("\n使用 np.dot(A, B) 的结果:\n", C_dot)
```

为了清晰和易读，最好将 `@` 用于矩阵乘法，而 `np.dot()` 则用于明确计算两个向量的点积。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本经典的线性代数教材，为矩阵运算（包括乘法及其性质）提供了扎实的理论基础。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  涵盖线性代数概念，包括矩阵乘法，特别侧重于它们在机器学习算法中的应用和相关性。
- [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) — NumPy Developers (2023)
  Publisher: NumPy Project
  NumPy 矩阵乘法函数的官方文档，详细说明了其用法和 `@` 运算符，对实际实现至关重要。
