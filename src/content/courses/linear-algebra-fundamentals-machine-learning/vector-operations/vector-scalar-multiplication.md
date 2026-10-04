---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "vector-operations"
lesson: "vector-scalar-multiplication"
sourceId: 7809
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-2-vector-operations/vector-scalar-multiplication"
title: "标量乘法"
description: "明白将向量乘以标量如何改变其大小和/或方向。"
order: 2
plots: ["plots/7809-0.json"]
sourceHash: "e24751630cae805c0ee8fa708f38e678a16349e49f2b3ee30cb34dc63f8d4ece"
sourceCorrections: []
---

在通过加法和减法组合向量 (vector)之后，下一个基本运算是缩放它们。标量乘法使我们能够在不改变向量方向的情况下拉伸或收缩向量，这在许多机器学习 (machine learning)算法中非常常用。可以将其视为调整数据点特征的强度或大小。

### 缩放的代数原理

本质上，标量乘法很简单。您取一个数字，称为**标量**，并将其与一个向量 (vector)相乘。该运算是按元素进行的，这意味着您将标量乘以向量中的每个元素。

标量乘法是线性代数中的一项基本运算。它能够拉伸或收缩向量而不改变其方向。在机器学习 (machine learning)的许多算法中，这一过程常用于调整数据点特征的强度或大小。

### 几何含义：拉伸与收缩

标量乘法真正的直观感觉来自对其的视觉呈现。将向量 (vector)乘以标量会改变其长度，即**大小**。标量的符号决定了向量的方向是保持不变还是反转。

- **拉伸：** 如果标量的绝对值大于1（例如，$c = 2$ 或 $c = -3$），向量会变长。
- **收缩：** 如果标量的绝对值在0到1之间（例如，$c = 0.5$ 或 $c = -0.2$），向量会变短。
- **方向反转：** 如果标量是负数（例如，$c = -1$），向量的方向会完全反转。它指向与原始方向相反180度。

下图显示了将向量 $\vec{v}$ 乘以不同标量的效果。



![Interactive chart](plots/7809-0.json)



> 向量 $\vec{v}$ 被不同的因子缩放。乘以2会拉伸它，乘以0.5会收缩它，乘以-1.5则会反转其方向并拉伸它。

### 在NumPy中实现标量乘法

在Python中，NumPy使标量乘法用起来很自然。您可以在数字和NumPy数组之间使用标准的乘法运算符 (`*`)，NumPy会自动处理按元素进行的运算。

```python
import numpy as np

# 定义一个NumPy数组形式的向量
v = np.array([3, 1])
print(f"原始向量 v: {v}")

# 乘以标量 c = 2
c = 2
scaled_v = c * v
print(f"向量 v 被 {c} 缩放后: {scaled_v}")

# 乘以标量 c = -1.5
c = -1.5
flipped_v = c * v
print(f"向量 v 被 {c} 缩放后: {flipped_v}")
```

执行此代码将产生以下输出：

```
Original vector v: [3 1]
Vector v scaled by 2: [6 2]
Vector v scaled by -1.5: [-4.5 -1.5]
```

如您所见，代码简洁明了，直接反映了数学运算。无需自己遍历向量 (vector)元素；NumPy的优化运算能高效地处理它。

### 实用案例：学习率

标量乘法不仅仅是一个抽象运算；它是机器学习 (machine learning)中的一个重要工具。其中一个最常见的应用是在**梯度下降 (gradient descent)**等优化算法中。

训练模型时，梯度下降会计算一个 `梯度` 向量 (vector)，该向量指向误差增加最陡峭的方向。为了最小化误差，我们需要朝*相反*的方向移动。但我们应该移动多远？这就是**学习率**发挥作用的地方。

学习率是一个小的标量（例如 0.01），它对负梯度向量进行缩放。模型权重 (weight)的更新规则通常如下所示：


$$
\text{新权重} = \text{旧权重} - \text{学习率} \times \text{梯度}
$$


这里，`学习率` $\times$ `梯度` 就是一次标量乘法。它会收缩梯度向量，以确保我们朝着正确的方向迈出一小步，小心前进。过大的学习率可能会导致我们跳过最优解，而过小的学习率会使训练过程太慢。标量乘法是赋予我们这种精细控制的运算。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebras/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  这本基础教科书全面涵盖了向量空间和运算，包括标量乘法，提供了严谨的数学定义和几何直观理解。
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — Stephen Boyd and Lieven Vandenberghe (2018)
  一本非常好的资源，强调线性代数的应用，与机器学习从业者的计算和分析需求直接相关。从应用角度介绍了向量运算。
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html) — NumPy Developers (2025)
  NumPy 的官方文档详细说明了数组对象及其操作，包括标量乘法如何在 Python 中高效实现，用于数值计算。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2: Linear Algebra
  这本广受认可的教材包含专门的线性代数章节，解释了基本的向量运算及其与梯度下降等算法的直接相关性，其中标量乘法控制着学习率。
