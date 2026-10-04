---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "vector-operations"
lesson: "vector-dot-product"
sourceId: 7810
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-2-vector-operations/vector-dot-product"
title: "点积"
description: "学习计算两个向量的点积，并理解其在投影和角度方面的含义。"
order: 3
plots: []
sourceHash: "1bbb031f899191b300cb1d1f10929ba92cc29b4df15122e17e3cfebfa71ffef3"
sourceCorrections: []
---

在学习了如何对向量 (vector)进行加、减和标量乘法后，我们现在转向一种不同类型的运算：点积。与之前产生另一个向量的运算不同，点积接受两个向量，并生成一个单一的数值，即一个标量。这个简单的数值提供了极其有用的信息，因为它告诉我们两个向量方向之间的关联。它是机器学习 (machine learning)中一项最基本的运算，可用于从测量相似度到执行几何变换的各种场景。

### 代数计算

从其核心来看，点积是一个直接的计算。您将两个向量 (vector)的对应分量相乘，然后将结果相加。对于相同维度 $n$ 的两个向量 $v$ 和 $w$，它们的点积，表示为 $v \cdot w$，计算方式如下：


$$
v \cdot w = v_1 w_1 + v_2 w_2 + \dots + v_n w_n
$$


让我们以两个简单的二维向量为例：$v = [2, 3]$ 和 $w = [4, 1]$。

为了求它们的点积，我们将第一个分量相乘，第二个分量相乘，然后将乘积相加：

$v \cdot w = (2 \times 4) + (3 \times 1) = 8 + 3 = 11$

结果是11，一个标量值。只要两个向量具有相同数量的分量，此计算就适用于任何维度的向量。

### 几何含义

点积的真正效用来自于其几何解释。点积的值与两个向量 (vector)之间的夹角直接关联。连接它们的公式是：


$$
v \cdot w = \|v\| \|w\| \cos(\theta)
$$


此处，$\|v\|$ 和 $\|w\|$ 是向量 $v$ 和 $w$ 的大小（或范数），而 $\theta$ (theta) 是它们之间的夹角。此等式表明，点积本质上衡量了一个向量在多大程度上“指向”另一个向量的方向。

这种关联使我们能够解释点积结果的符号：

- **正点积 ($v \cdot w > 0$):** 向量之间的夹角 $\theta$ 小于 90°。向量大致指向相似的方向。
- **零点积 ($v \cdot w = 0$):** 夹角 $\theta$ 恰好是 90°。向量彼此垂直。这是一项特殊且非常有用的特性，称为**正交性**。
- **负点积 ($v \cdot w < 0$):** 夹角 $\theta$ 大于 90°。向量大致指向相反的方向。

以下图表说明了向量 $v$ 在向量 $w$ 上的投影。此投影的长度与点积有关。

> 点积与一个向量在另一个向量上的投影长度有关。同方向上更长的投影长度对应更大的正点积。

### 应用：测量相似度

这种几何含义使点积成为测量相似度的根本，这是机器学习 (machine learning)中一项常见的任务。通过重新排列几何公式，我们可以分离出夹角：


$$
\cos(\theta) = \frac{v \cdot w}{\|v\| \|w\|}
$$


此公式计算两个向量 (vector)之间的**余弦相似度**。它给出一个介于-1和1之间的值，为比较提供了一个清晰的衡量标准：

- **1:** 向量指向完全相同的方向 ($\theta = 0°$)。它们被认为是完全相似的。
- **0:** 向量是正交的 ($\theta = 90°$)。它们没有相似度。
- **-1:** 向量指向相反的方向 ($\theta = 180°$)。它们是完全不相似的。

例如，在自然语言处理中，文档可以转换为向量，其中每个分量代表一个词的频率。通过计算这些向量之间的余弦相似度，我们可以确定两篇文档在其主题内容上的相似程度。

### 使用NumPy计算点积

NumPy提供了高效的函数来执行线性代数运算。有两种常见的方法来计算点积。

第一种是使用 `numpy.dot()` 函数。

```python
import numpy as np

# 将两个向量定义为NumPy数组
v = np.array([2, 3])
w = np.array([4, 1])

# 使用 np.dot() 计算点积
dot_product = np.dot(v, w)

print(f"向量 v: {v}")
print(f"向量 w: {w}")
print(f"点积: {dot_product}")
```

运行此代码将产生预期输出：

```
向量 v: [2 3]
向量 w: [4 1]
点积: 11
```

一种更现代、更符合Python风格的方法来计算点积（以及执行矩阵乘法，我们稍后会看到）是使用 `@` 运算符。引入此运算符是为了使数值代码更清晰、更易读。

```python
import numpy as np

v = np.array([2, 3])
w = np.array([4, 1])

# 使用 @ 运算符计算点积
dot_product_operator = v @ w

print(f"使用 @ 运算符的点积: {dot_product_operator}")
```

此代码产生相同的结果，即 `11`。`@` 运算符常因其简洁性和明确的数学意图而受到青睐。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  全面直观地解释了点积，包括其代数和几何解释，以及它在理解向量关系中的作用。（第五版）
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  在机器学习背景下讨论了点积，强调了它对于相似性度量和其他基本操作的重要性。
- [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) — NumPy Developers (2023)
  NumPy 点积函数的官方文档，详细说明了其用法和行为。
- [Array objects: The N-dimensional array (ndarray) - Matrix multiplication](https://numpy.org/doc/stable/reference/arrays.ndarray.html#matrix-multiplication) — NumPy Developers (2023)
  解释了用于数组乘法的 `@` 运算符，对于一维数组，它执行点积，提供了简洁易读的语法。
