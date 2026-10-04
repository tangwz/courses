---
course: "linear-algebra-essentials-ml"
chapter: "solving-linear-systems-matrix-inverses"
lesson: "solving-axb-using-inverse"
sourceId: 1487
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses/solving-axb-using-inverse"
title: "运用逆矩阵求解 Ax=b"
description: "应用逆矩阵在方程 Ax=b 中求得向量 x 的解。"
order: 6
plots: []
sourceHash: "e07074d8fa2ba112cced399bd6e936f46a3f913f2214ac277fcf71b7e527b376"
sourceCorrections: []
---

对于可逆矩阵，可以计算出矩阵逆 $A^{-1}$。这个矩阵逆提供了一种直接求解基本线性系统 $Ax = b$ 的方法。

请记住，方程 $Ax = b$ 表示将线性变换 $A$ 作用于未知向量 (vector) $x$，得到已知向量 $b$。我们的目标是求得原始向量 $x$。如果矩阵 $A$ 是方阵且可逆，我们可以将其逆矩阵 $A^{-1}$ 视为一种“抵消” $A$ 作用的变换。

考虑方程组：
$Ax = b$

假设 $A$ 可逆，我们可以对方程两边同时左乘 $A^{-1}$。重要的是，为了保持等式，必须在两边都从*左侧*相乘，因为矩阵乘法通常不满足交换律。

$A^{-1}(Ax) = A^{-1}b$

运用矩阵乘法的结合律，我们可以将左侧的项重新分组：

$(A^{-1}A)x = A^{-1}b$

根据矩阵逆的定义，我们知道 $A^{-1}A = I$，其中 $I$ 是单位矩阵：

$Ix = A^{-1}b$

最后，由于任何向量乘以单位矩阵都保持不变（$Ix = x$），我们得到了 $x$ 的解：

$x = A^{-1}b$

这个漂亮的结论告诉我们，如果我们能求得矩阵 $A$ 的逆矩阵，我们就可以简单地通过将 $A^{-1}$ 乘以向量 $b$ 来求得解向量 $x$。

## 运用逆矩阵方法的条件

此方法完全取决于矩阵逆 $A^{-1}$ 的存在。因此，它仅适用于以下情况：

1. **A 是方阵：** 方程的数量必须等于未知数的数量（即 $A$ 必须是 $n \times n$）。逆矩阵只对方阵有定义。
2. **A 是可逆（非奇异）的：** $A$ 的行列式必须非零（$\det(A) \neq 0$）。如果行列式为零，则矩阵没有逆矩阵。在这种情况下，方程组 $Ax=b$ 要么无解，要么有无穷多解，并且此方法无法求得唯一解。

如果满足这些条件，则方程组 $Ax=b$ 有唯一解，由 $x = A^{-1}b$ 给出。

## 示例：求解一个简单方程组

让我们将此应用于一个简单的 2x2 方程组：


$$
\begin{align*}
2x_1 + 3x_2 &= 8 \\
1x_1 + 4x_2 &= 9
\end{align*}
$$


用矩阵形式表示，即 $Ax = b$，其中：


$$
A = \begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix}, \quad x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad b = \begin{bmatrix} 8 \\ 9 \end{bmatrix}
$$


首先，我们通过计算 $A$ 的行列式来检查它是否可逆：
$\det(A) = (2)(4) - (3)(1) = 8 - 3 = 5$。由于行列式是 5（非零），所以逆矩阵存在。

在上一节（“计算矩阵的逆”）中，我们可能已经得出（或者现在可以使用 2x2 逆矩阵的公式计算出）：


$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 4 & -3 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 4/5 & -3/5 \\ -1/5 & 2/5 \end{bmatrix}
$$


现在，我们可以运用公式 $x = A^{-1}b$ 求得 $x$：


$$
x = \begin{bmatrix} 4/5 & -3/5 \\ -1/5 & 2/5 \end{bmatrix} \begin{bmatrix} 8 \\ 9 \end{bmatrix}
$$


执行矩阵-向量 (vector)乘法：


$$
x = \begin{bmatrix} (4/5)(8) + (-3/5)(9) \\ (-1/5)(8) + (2/5)(9) \end{bmatrix} = \begin{bmatrix} (32/5) - (27/5) \\ (-8/5) + (18/5) \end{bmatrix} = \begin{bmatrix} 5/5 \\ 10/5 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$


因此，解为 $x_1 = 1$ 和 $x_2 = 2$。我们可以快速验证这一点：$2(1) + 3(2) = 2+6 = 8$ 和 $1(1) + 4(2) = 1+8 = 9$。解是正确的。

## NumPy 实现

尽管手动计算对于小型矩阵有指导意义，但对于机器学习 (machine learning)中遇到的系统，我们通常依赖于 NumPy 这样的数值计算库。NumPy 提供了计算逆矩阵和高效执行必要矩阵乘法的函数。

```python
import numpy as np

# 定义示例中的矩阵 A 和向量 b
A = np.array([[2, 3],
              [1, 4]])

b = np.array([[8],   # 将 b 定义为列向量 (2x1)
              [9]])

# 检查 A 是否为方阵
rows, cols = A.shape
if rows != cols:
    print("矩阵 A 必须是方阵才能求逆。")
else:
    # 计算行列式
    det_A = np.linalg.det(A)
    print(f"A 的行列式：{det_A:.2f}")

    # 检查行列式是否接近于零（在机器精度范围内）
    if np.isclose(det_A, 0):
        print("矩阵 A 是奇异的（或近似奇异的）。")
        print("无法运用逆矩阵方法求解。")
    else:
        # 使用 np.linalg.inv() 计算 A 的逆矩阵
        try:
            A_inv = np.linalg.inv(A)
            print("\nA 的逆矩阵 (A^{-1})：")
            print(A_inv)

            # 计算解 x = A_inv * b
            # 使用 @ 运算符进行矩阵乘法
            x = A_inv @ b
            # 或者：x = np.dot(A_inv, b)

            print("\n解向量 x = A^{-1}b：")
            print(x)

            # 验证：检查 A @ x 是否接近原始 b
            print("\n验证 (A @ x)：")
            print(A @ x)
            print(f"\nA @ x 是否接近 b？{np.allclose(A @ x, b)}")

        except np.linalg.LinAlgError:
             # 处理逆矩阵计算在数值上失败的情况
             print("NumPy 在计算逆矩阵时遇到错误。")
```

> **Python 代码输出：**
>
> ```
> A 的行列式： 5.00
>
> A 的逆矩阵 (A^{-1})：
> [[ 0.8 -0.6]
>  [-0.2  0.4]]
>
> 解向量 x = A^{-1}b：
> [[1.]
>  [2.]]
>
> 验证 (A @ x)：
> [[8.]
>  [9.]]
>
> A @ x 是否接近 b？ True
> ```

这段代码运用 NumPy 复制了我们的手动计算步骤。它首先检查 `A` 是否为方阵，然后使用 `np.linalg.det()` 计算行列式。如果行列式非零，它会使用 `np.linalg.inv()` 计算逆矩阵，然后通过 `@` 运算符将 `A_inv` 与 `b` 相乘来求得解 `x`。`np.allclose()` 函数对于验证结果很有用，它考虑了潜在的小浮点误差。

## 关于实际计算的说明

尽管运用显式逆矩阵 $x = A^{-1}b$ 求解 $Ax=b$ 清晰且数学上正确，但在实际中，它通常*不是*最有效或数值最稳定的方法，特别是对于机器学习 (machine learning)应用中常见的大型矩阵。

1. **计算成本：** 计算逆矩阵 $A^{-1}$ 通常比高斯消元法或 LU 分解等方法需要更多的计算操作（浮点运算），这些方法专门设计用于直接求解系统 $Ax=b$。
2. **数值稳定性：** 计算机算术使用有限精度（浮点数）。计算逆矩阵的过程有时会放大矩阵 $A$ 中存在的微小精度误差。直接求解方法通常旨在最大程度地减少此类错误的累积，从而得到更准确的解。

基于这些原因，NumPy 和 SciPy 等库提供了诸如 `np.linalg.solve(A, b)` 的函数。该函数直接求解 $Ax=b$ 中的 $x$，通常使用高效且稳定的底层算法（如 LU 分解）。它避免计算完整的逆矩阵 $A^{-1}$。

```python
# np.linalg.solve 的使用示例（通常更受偏爱）
try:
    x_solve = np.linalg.solve(A, b)
    print("\n运用 np.linalg.solve(A, b) 得到的解：")
    print(x_solve)
    print(f"从 solve() 得到的解是否接近 b？{np.allclose(A @ x_solve, b)}")
except np.linalg.LinAlgError:
    print("\n矩阵 A 是奇异的。无法使用 np.linalg.solve 求解。")
```

> **np.linalg.solve 示例的输出：**
>
> ```
> 运用 np.linalg.solve(A, b) 得到的解：
> [[1.]
>  [2.]]
> 从 solve() 得到的解是否接近 b？ True
> ```

当你的主要目标只是为了求得解向量 (vector) $x$ 时，通常应优先使用 `np.linalg.solve(A, b)` 而不是 `np.linalg.inv(A) @ b`。

然而，理解逆矩阵方法仍然很重要。它为线性系统提供了基本的理论理解，出现在许多解析推导中（例如，在线性回归的 Normal Equations 的推导中），并且非常适合较小的系统或需要逆矩阵 $A^{-1}$ 本身进行进一步分析的情况。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本基础教材，全面涵盖矩阵逆、线性系统求解以及可逆性的理论条件。
- [Numerical Linear Algebra](https://doi.org/10.1137/1.9781611971212) — Lloyd N. Trefethen and David Bau III (1997)
  Publisher: SIAM (Society for Industrial and Applied Mathematics); DOI: [10.1137/1.9781611971212](https://doi.org/10.1137/1.9781611971212)
  数值线性代数的基础著作，深入探讨求解线性系统算法的计算效率和稳定性，并将直接方法与矩阵求逆进行对比。
- [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) — NumPy Developers (2024)
  NumPy 计算矩阵逆的官方文档，提供关于其用法、参数以及奇异矩阵错误处理的详细信息。
- [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) — NumPy Developers (2024)
  NumPy 直接求解线性系统 $Ax=b$ 的官方文档，解释了其在计算性能和数值稳定性方面优于显式矩阵求逆方法的优势。
