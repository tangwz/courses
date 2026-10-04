---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "matrix-operations"
lesson: "matrix-addition-subtraction"
sourceId: 7814
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-3-matrix-operations/matrix-addition-subtraction"
title: "矩阵加法和减法"
description: "介绍矩阵的加法和减法，包括其维度规则。"
order: 1
plots: []
sourceHash: "2afd7cd3fc164da2a00cc246f03be97e0f0600dcdef5ec89c832780ad1920974"
sourceCorrections: []
---

正如我们能对单个数字和向量 (vector)进行运算一样，我们也能对矩阵进行运算。其中最简单的运算是加法和减法。这些运算对于需要组合或比较具有相同结构的数据集任务来说非常基础。例如，如果你有两个矩阵表示两个连续月份的销售数据，将它们相加就能得到该期间的总销售额。

### 第一条规则：维度必须一致

在对两个矩阵进行加法或减法运算之前，你必须检查一个必要条件：**矩阵的维度必须完全相同**。这意味着它们必须具有相同的行数和列数。

如果你有一个维度为 $m \times n$（读作“m乘n”，表示 $m$ 行 $n$ 列）的矩阵 $A$ 和一个维度为 $p \times q$ 的矩阵 $B$，你只有在 $m = p$ 且 $n = q$ 时才能对它们进行加法或减法运算。如果它们的形状不完美对齐 (alignment)，则该运算是未定义的。你不能将一个没有对应伙伴的元素添加到另一个矩阵中。

### 运算方式：按元素操作

一旦你确认两个矩阵具有相同的维度，它们的加法或减法过程就简单直接。该运算是**按元素进行**的。这意味着你取第一个矩阵第一行第一列的元素，并将其与第二个矩阵第一行第一列的元素相加。你对每个位置都重复此操作。

假设我们有两个矩阵 $A$ 和 $B$，我们想找到一个新矩阵 $C$，使得 $C = A + B$。矩阵 $C$ 中位于第 $i$ 行第 $j$ 列的任意元素（表示为 $C_{ij}$）简单地就是 $A$ 和 $B$ 中相同位置元素的和。

形式上，这写为：


$$
C_{ij} = A_{ij} + B_{ij}
$$


同样的原则也适用于减法：


$$
C_{ij} = A_{ij} - B_{ij}
$$


下图说明了两个2x2矩阵的按元素相加。结果矩阵中的每个元素都是输入矩阵中对应位置元素的和。

> 结果矩阵中每个位置为 ($i, j$) 的元素，是原始矩阵中位置为 ($i, j$) 的元素的和。

### 一个实际例子

假设一家小公司记录了两名销售代表在三种产品类别中的销售数量。一月份的销售额存储在矩阵 $J$ 中，二月份的销售额存储在矩阵 $F$ 中。

以下是一月份的销售额（矩阵 $J$）：


$$
J = \begin{bmatrix} 50 & 35 & 110 \\ 75 & 90 & 60 \end{bmatrix}
$$


以及二月份的销售额（矩阵 $F$）：


$$
F = \begin{bmatrix} 45 & 40 & 100 \\ 80 & 110 & 50 \end{bmatrix}
$$


两个矩阵都是 $2 \times 3$，所以我们可以将它们相加，得到这两个月的总销售额。


$$
\text{总销售额} = J + F = \begin{bmatrix} 50+45 & 35+40 & 110+100 \\ 75+80 & 90+110 & 60+50 \end{bmatrix} = \begin{bmatrix} 95 & 75 & 210 \\ 155 & 200 & 110 \end{bmatrix}
$$


同样地，如果我们想查看从一月到二月的销售额变化，可以用 $F$ 减去 $J$：


$$
\text{销售额变化} = F - J = \begin{bmatrix} 45-50 & 40-35 & 100-110 \\ 80-75 & 110-90 & 50-60 \end{bmatrix} = \begin{bmatrix} -5 & 5 & -10 \\ 5 & 20 & -10 \end{bmatrix}
$$


结果矩阵显示，销售代表 1 的产品 1 和产品 3 的销售额下降，但产品 2 的销售额增加。销售代表 2 的所有产品销售额都增加了。

### 矩阵加法的性质

矩阵加法遵循标准算术中的一些熟悉性质，这些性质值得记住：

- **交换律：** 加法的顺序无关紧要。 $A + B = B + A$
- **结合律：** 当相加三个或更多矩阵时，分组方式无关紧要。 $(A + B) + C = A + (B + C)$

这些性质之所以成立，是因为其基础运算只是单个数字的加法，而数字加法本身就具有交换律和结合律。

### NumPy 中的矩阵加法

在 Python 中使用 NumPy 库时，执行这些运算就像使用标准的 `+` 和 `-` 运算符一样简单。NumPy 会自动处理按元素计算。

```python
import numpy as np

# 一月份销售额
J = np.array([
    [50, 35, 110],
    [75, 90, 60]
])

# 二月份销售额
F = np.array([
    [45, 40, 100],
    [80, 110, 50]
])

# 总销售额
total_sales = J + F
print("总销售额 (J + F):")
print(total_sales)

# 销售额变化
change_in_sales = F - J
print("\n销售额变化 (F - J):")
print(change_in_sales)
```

如果你在 NumPy 中尝试对维度不匹配的矩阵进行加法或减法运算，它将抛出 `ValueError`，防止你执行无效运算。这种行为在实际编程环境中强调了数学规则。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本广受认可的线性代数教材，全面讲解了包括加减法在内的基本矩阵运算。
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](http://vmlsbook.stanford.edu/) — Stephen Boyd and Lieven Vandenberghe (2018)
  Publisher: Cambridge University Press
  这本书从应用角度讲解线性代数，与机器学习背景高度相关，并详细解释了矩阵运算。
- [Array objects - Basic operations](https://numpy.org/doc/stable/user/basics.html#basic-operations) — NumPy Developers (2024)
  NumPy的官方文档，提供了在Python中使用数组对象实现矩阵加法和减法的细节和示例。
