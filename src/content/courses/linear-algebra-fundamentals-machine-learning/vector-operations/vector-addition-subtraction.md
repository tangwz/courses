---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "vector-operations"
lesson: "vector-addition-subtraction"
sourceId: 7808
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-2-vector-operations/vector-addition-subtraction"
title: "向量加法与减法"
description: "学习向量加法和减法的代数和几何解释。"
order: 1
plots: []
sourceHash: "ded58fe37d08cc8b0f28f8bbc52da2b123b8578db884b9d539b29fc33d34d15e"
sourceCorrections: []
---

就像我们可以对单个数字（标量）进行算术运算一样，我们也可以对向量 (vector)进行算术运算。这些运算是机器学习 (machine learning)中许多算法的根本，从计算误差到调整模型参数 (parameter)。我们将从两个最直接的运算开始：加法和减法。

### 向量 (vector)加法的代数与几何

两个向量相加是一个逐元素操作。这表示你只需将每个向量的对应分量相加，就能得到新向量的分量。为了完成此操作，这些向量必须具有相同的维度，也就是说它们必须包含相同数量的分量。

假设我们有两个二维向量，$v$ 和 $w$：


$$
v = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}, \quad w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}
$$


它们的和，$v + w$，计算如下：


$$
v + w = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \end{bmatrix}
$$


举个具体例子，如果 $v = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ 且 $w = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$，它们的和是：


$$
v + w = \begin{bmatrix} 2 + 1 \\ 1 + 3 \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$


尽管代数运算简单，但其几何解释能提供更清晰的认识。为了直观地表示向量加法，我们使用**首尾相接法**。你从原点开始画第一个向量。然后，从第一个向量的末端（或头部）开始画第二个向量。这两个向量的和是那个从原点开始，到第二个向量末端结束的新向量。

> 向量加法的首尾相接法。得到的向量 $v+w$ 代表了两个向量的组合位移。

另一种理解方式是**平行四边形法则**。如果你从原点开始画向量 $v$ 和 $w$，它们会构成一个平行四边形的两条相邻边。它们的和 $v+w$ 是该平行四边形中同样从原点开始的对角线。注意，这得到相同的结果，因为 $v+w = w+v$。这被称为向量加法的交换律。

### 向量 (vector)减法：寻找差值

向量减法也像加法一样，是逐元素操作。要从向量 $v$ 中减去向量 $w$，你需要将 $w$ 的分量从 $v$ 的对应分量中减去。同样，这些向量必须具有相同的维度。


$$
v - w = \begin{bmatrix} v_1 - w_1 \\ v_2 - w_2 \end{bmatrix}
$$


使用我们之前的向量，$v = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ 和 $w = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$，减法运算如下：


$$
v - w = \begin{bmatrix} 3 - 1 \\ 4 - 3 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$


从几何角度看，理解减法最简单的方法是将其视为**加上相反向量**。向量 $-w$ 的长度与 $w$ 相同，但指向完全相反的方向。因此，运算 $v - w$ 等同于 $v + (-w)$。我们可以应用相同的首尾相接法则：画出向量 $v$，然后从其末端画出向量 $-w$。

> 向量减法可视为加上一个相反向量。向量 $v-w$ 是沿着 $v$ 移动后再沿着 $-w$ 移动的结果。

这个运算有助于找到从一个点指向另一个点的向量。从 $w$ 定义的点到 $v$ 定义的点的向量正好是 $v - w$。

### 使用 NumPy 在 Python 中进行向量 (vector)运算

手动执行这些操作有助于理解，但在实际中，我们使用 NumPy 这样的库来完成工作。NumPy 的数组非常适合表示向量，并且标准算术运算符（`+` 和 `-`）会自动执行逐元素加法和减法。

首先，让我们导入 NumPy 并创建两个向量。

```python
import numpy as np

v = np.array([2, 1])
w = np.array([1, 3])

print(f"Vector v: {v}")
print(f"Vector w: {w}")
```

现在，我们可以使用简单的 `+` 运算将它们相加。

```python
# 向量加法
sum_vw = v + w
print(f"v + w = {sum_vw}")
```

**输出：**

```
向量 v: [2 1]
向量 w: [1 3]
v + w = [3 4]
```

使用 `-` 运算符，减法同样简单。

```python
# 向量减法
diff_vw = sum_vw - w
print(f"[3, 4] - w = {diff_vw}")
```

**输出：**

```
[3, 4] - w = [2 1]
```

如你所见，NumPy 为你处理了逐分量算术运算。这不仅方便，而且在性能方面进行了高度优化，特别是在处理具有数千甚至数百万维度的向量时，这在机器学习 (machine learning)中很常见。
