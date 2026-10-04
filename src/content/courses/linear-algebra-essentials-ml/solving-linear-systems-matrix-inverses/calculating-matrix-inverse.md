---
course: "linear-algebra-essentials-ml"
chapter: "solving-linear-systems-matrix-inverses"
lesson: "calculating-matrix-inverse"
sourceId: 1483
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses/calculating-matrix-inverse"
title: "计算矩阵的逆"
description: "计算小矩阵逆矩阵的方法，以及计算大矩阵逆矩阵的计算工具使用。"
order: 4
plots: []
sourceHash: "8303c3caebe819016dbec206cdc6f7cca96a00573ddea09048dffe0dbf64187a"
sourceCorrections: []
---

方阵 $A$ 的逆矩阵 $A^{-1}$ 是求解线性方程组的核心。当 $A^{-1}$ 存在时，方程组 $Ax = b$ 可以通过计算 $x = A^{-1}b$ 来求解。那么，我们如何才能实际找到这个矩阵 $A^{-1}$ 呢？计算 $A^{-1}$ 的方法很大程度上取决于矩阵的大小以及是手动计算还是使用计算机。

### 2x2 矩阵的情况：一个简便的公式

对于 2x2 矩阵，有一个直接的公式。假设我们的矩阵是：


$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$


它的逆矩阵 $A^{-1}$ 由以下公式给出：


$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$


请注意分母中的项 $ad - bc$。这是 2x2 矩阵 $A$ 的*行列式*，我们将在下一节对此进行更多讨论。要使逆矩阵存在，此行列式*必须不为零*。如果 $ad - bc = 0$，则该矩阵是奇异的（或不可逆的），且 $A^{-1}$ 不存在。

该公式主要包括交换对角线元素（$a$ 和 $d$），对非对角线元素（$-b$ 和 $-c$）取负，并将整个结果矩阵除以行列式。

**示例：**
计算 $A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}$ 的逆矩阵。

1. **计算行列式：** $ad - bc = (4)(6) - (7)(2) = 24 - 14 = 10$。由于 $10 \neq 0$，逆矩阵存在。
2. **应用公式：**
   
   $$
   A^{-1} = \frac{1}{10} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 6/10 & -7/10 \\ -2/10 & 4/10 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}
   $$
   

您可以通过将 $A$ 和 $A^{-1}$ 相乘（以任意顺序）来验证这一点，看是否得到单位矩阵 $I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$。

### 更大的矩阵：高斯-约旦消元法

对于大于 2x2（有时甚至是 3x3）的矩阵，手动使用公式（例如纯数学课程中讲授的伴随矩阵法）计算逆矩阵会变得极其繁琐且容易出错。标准的算法方法，也是软件中通常实现的方法，是基于**高斯-约旦消元法**。

其思路是将矩阵 $A$ 与同等大小的单位矩阵 $I$ 增广，创建 $[A | I]$。然后，应用初等行变换（与高斯消元法求解方程组所用的操作相同），将左侧（$A$）转换为单位矩阵 $I$。奇妙之处在于，*相同的一系列*行变换将把右侧（$I$）转换为逆矩阵 $A^{-1}$。最终形式将是 $[I | A^{-1}]$。

初等行变换包括：

1. 交换两行。
2. 将某行乘以一个非零标量。
3. 将某行的一个倍数加到另一行。

了解此过程固然有益，但在机器学习 (machine learning)情境下，您几乎*从不*会手动执行此操作。它计算量大，最好留给优化过的库来处理。

### 使用 NumPy 进行矩阵求逆

实际上，尤其是在使用 Python 的机器学习 (machine learning)工作流中，您会依赖 NumPy 等数值计算库来计算矩阵的逆。NumPy 的线性代数模块 `numpy.linalg` 提供了 `inv()` 函数。

以下是如何使用 NumPy 计算我们之前 2x2 示例矩阵的逆：

```python
import numpy as np

# 定义矩阵 A
A = np.array([[4, 7],
              [2, 6]])

# 计算逆矩阵
try:
    A_inv = np.linalg.inv(A)
    print("矩阵 A:")
    print(A)
    print("\n矩阵 A 的逆：")
    print(A_inv)

    # 验证（可选）：将 A 乘以 A_inv
    # 结果应为单位矩阵（在浮点精度范围内）
    identity_check = np.dot(A, A_inv)
    print("\n验证 (A @ A_inv)：")
    # 使用 np.allclose 处理可能的小浮点误差
    print(np.round(identity_check, decimals=5))
    print("\n是否接近单位矩阵：", np.allclose(identity_check, np.eye(2)))

except np.linalg.LinAlgError:
    print("\n矩阵 A 是奇异的，无法求逆。")
```

**输出：**

```
矩阵 A:
[[4 7]
 [2 6]]

矩阵 A 的逆：
[[ 0.6 -0.7]
 [-0.2  0.4]]

验证 (A @ A_inv):
[[1. 0.]
 [0. 1.]]

是否接近单位矩阵: True
```

这与我们的手动计算结果相符。

对于更大的矩阵：

```python
# 3x3 矩阵示例
B = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# 计算逆矩阵
try:
    B_inv = np.linalg.inv(B)
    print("\n矩阵 B:")
    print(B)
    print("\n矩阵 B 的逆：")
    print(B_inv)

    # 验证
    identity_check_B = np.dot(B, B_inv)
    print("\n验证 (B @ B_inv 是否接近单位矩阵):", np.allclose(identity_check_B, np.eye(3)))

except np.linalg.LinAlgError:
    print("\n矩阵 B 是奇异的，无法求逆。")
```

**输出：**

```
矩阵 B:
[[1 2 3]
 [0 1 4]
 [5 6 0]]

矩阵 B 的逆：
[[-24.  18.   5.]
 [ 20. -15.  -4.]
 [ -5.   4.   1.]]

验证 (B @ B_inv 是否接近单位矩阵): True
```

NumPy 高效地处理底层的高斯-约旦消元法（或其数值稳定的变体）。如果您将奇异矩阵（行列式为零的矩阵）传递给 `np.linalg.inv()`，它将抛出 `LinAlgError`。

在尝试计算逆矩阵之前，通常最好先检查矩阵是否可逆。这就引出了行列式。

## 参考资料

- [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) — NumPy Developers ()
  Publisher: NumPy Project
  NumPy `linalg.inv`函数的官方文档，详细说明了其在计算矩阵逆时的用法、对奇异矩阵错误的处理以及Python中的实际实现示例。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  本书将线性代数概念（包括矩阵求逆）置于机器学习的背景下，提供了其在机器学习算法中实际应用和理论重要性的视角。
