---
course: "linear-algebra-essentials-ml"
chapter: "vector-spaces-subspaces-linear-independence"
lesson: "linear-combinations-span"
sourceId: 1499
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-4-vector-spaces-subspaces-linear-independence/linear-combinations-span"
title: "线性组合与张成"
description: "了解向量如何组合以及一组向量的张成空间。"
order: 3
plots: ["plots/1499-0.json"]
sourceHash: "f8a01dbad6eaa5a79f410ea3e2c57756d8351980b26b1ea27dc70f6455ca7c8a"
sourceCorrections: []
---

在前面几章中，我们了解了向量 (vector)如何表示数据点以及矩阵如何变换它们。现在，让我们看看如何组合向量以生成新向量。这一过程对理解向量空间很基本，它被称为形成**线性组合**。

一组向量 $v_1, v_2, ..., v_k$ 的线性组合是任何可以表示成以下形式的向量 $w$:


$$
w = c_1 v_1 + c_2 v_2 + ... + c_k v_k
$$


其中 $c_1, c_2, ..., c_k$ 是标量常数（实数）。简单来说，你取一组向量，将每个向量按一定量（可能为零或负数）进行缩放，然后将这些缩放后的向量相加。结果就是线性组合。

在二维空间中从几何角度思考。如果你有一个向量 $v_1$，它的线性组合就是 $v_1$ 的标量倍数，这些组合形成一条通过原点并沿着 $v_1$ 方向的直线。

如果你在 $\mathbb{R}^2$ 中有两个向量，比如 $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 和 $v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$，你能生成哪些向量？
线性组合看起来像 $w = c_1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + c_2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 \end{bmatrix}$。通过选择不同的标量 $c_1$ 和 $c_2$，你可以得到二维平面上的*任何*点（向量）。

### 向量 (vector)的张成空间

这就引出了**张成空间**。一组向量 $\{v_1, v_2, ..., v_k\}$ 的张成空间是这些向量的*所有可能*线性组合的集合。它表示通过组合这些向量可以“到达”或生成的所有区域或空间。

- **单个非零向量的张成空间：** 通过原点的一条直线。
- **$\mathbb{R}^2$中两个不共线向量的张成空间：** 整个二维平面（$\mathbb{R}^2$）。
- **$\mathbb{R}^2$中两个共线向量的张成空间：** 通过原点的一条直线（与任一向量单独定义的直线相同）。
- **$\mathbb{R}^3$中两个不共线向量的张成空间：** 3D空间中通过原点的一个平面。
- **$\mathbb{R}^3$中三个不共面向量的张成空间：** 整个3D空间（$\mathbb{R}^3$）。

让我们在二维空间中可视化两个不共线向量的张成空间。考虑 $v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ 和 $v_2 = \begin{bmatrix} -1 \\ 3 \end{bmatrix}$。它们的张成空间是整个 $\mathbb{R}^2$。$\mathbb{R}^2$中的任何向量，例如 $w = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$，都可以写成线性组合 $c_1 v_1 + c_2 v_2$。



![R^2中两个向量的张成空间](plots/1499-0.json)



> 两个不共线向量 $v_1$ 和 $v_2$。平面中的任何向量 $w$（如绿色虚线向量）都可以通过对 $v_1$ 和 $v_2$ 进行缩放和相加得到。虚线表示通过原点并沿着 $v_1$ 和 $v_2$ 方向的无限延长线。张成空间填满了整个二维平面。

### 张成空间在机器学习 (machine learning)中的作用

在机器学习中，特征通常形成向量 (vector)。数据集可以看作是这些特征向量的集合。张成空间的思想帮助我们理解特征的“覆盖范围”。

- **特征空间：** 你的特征向量的张成空间定义了模型实际操作的特征空间。如果新的数据点超出训练数据特征的张成空间，模型可能难以泛化。
- **冗余：** 如果一个特征向量位于其他向量的张成空间内，它可能是冗余的。它不会为其他向量定义的空间增加新的“方向”或信息。这与线性独立性很相关，我们将在接下来讨论。
- **维度：** 张成特定空间所需的向量数量与其维度相关。如果你的数据存在于 $\mathbb{R}^{100}$ 中，但特征向量只张成一个10维子空间，这表明有维度缩减的可能。

### 使用NumPy计算线性组合

NumPy使得计算线性组合变得简单。

```python
import numpy as np

# 定义向量
v1 = np.array([2, 1])
v2 = np.array([-1, 3])

# 定义标量
c1 = 2.1538 # Approximately 28/13
c2 = 0.3077 # Approximately 4/13

# 计算线性组合 w = c1*v1 + c2*v2
w = c1 * v1 + c2 * v2

print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")
print(f"标量 c1={c1:.4f}, c2={c2:.4f}")
print(f"线性组合 w: {w}")
# 由于浮点表示，结果应接近 [4, 5]
# 让我们检查 w 是否确实是 [4, 5]
target_w = np.array([4, 5])
print(f"w是否近似等于 [4, 5]？ {np.allclose(w, target_w)}")
```

这段代码演示了如何将向量 (vector) $w = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$ 表示为 $v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ 和 $v_2 = \begin{bmatrix} -1 \\ 3 \end{bmatrix}$ 的线性组合。我们找到了实现这一目标的特定标量 $c_1 \approx 2.15$ 和 $c_2 \approx 0.31$。（寻找这些标量需要解线性方程组，这是第三章讨论过的一个内容）。

理解线性组合和张成空间提供了一个几何和代数框架，用于思考向量如何相互关联以及它们生成的空间。这个基础很有用，因为我们接下来将分析向量的独立性以及特征空间的结构。
