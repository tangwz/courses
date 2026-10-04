---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "identity-matrix"
sourceId: 7822
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/identity-matrix"
title: "单位矩阵"
description: "理解单位矩阵在矩阵代数中作为乘法单位元的作用。"
order: 2
plots: []
sourceHash: "6c32b614c8903d5baea8cd87c083d1ac6d781717ef865c77b2f6a2056af49de0"
sourceCorrections: []
---

数字1在算术中有一个特殊性质：任何数乘以1，结果保持不变。例如，$5 \times 1 = 5$。这个数被称为乘法单位元。矩阵代数也有类似的想法，它被称为**单位矩阵**。

单位矩阵，通常记作$I$，是一个方阵（行数和列数相等），其主对角线上的元素是1，其他所有元素都是0。主对角线从左上角延伸到右下角。

以下是2x2和3x3的单位矩阵：


$$
I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$


$$
I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$


下标，例如$I_2$中的2，表示矩阵的大小。通常，大小由具体情况决定，所以我们只写$I$。

### 乘法单位元性质

单位矩阵的决定性特点是，当你用它乘以任何矩阵$A$时，结果仍然是$A$，没有变化。乘法必须是有效的，这意味着维度必须匹配。对于任何$n \times m$矩阵$A$：


$$
A I_m = A
$$


$$
I_n A = A
$$


让我们用一个2x2矩阵来观察这一点。假设我们有矩阵$A$：


$$
A = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix}
$$


当我们用2x2单位矩阵$I_2$乘以$A$时：


$$
A I = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} (2 \cdot 1 + 3 \cdot 0) & (2 \cdot 0 + 3 \cdot 1) \\ (4 \cdot 1 + 5 \cdot 0) & (4 \cdot 0 + 5 \cdot 1) \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix}
$$


正如你所看到的，结果是原始矩阵$A$。

### 单位矩阵对方程求解有何作用

那么，这如何帮助我们解决本章的主要问题$Ax = b$呢？

回想一下你是如何解像$5x = 10$这样的简单代数方程的。你通过将方程两边乘以5的倒数，即$\frac{1}{5}$来分离出$x$。这使得$x$的系数变为1：


$$
(\frac{1}{5} \cdot 5)x = \frac{1}{5} \cdot 10
$$


简化后得到：


$$
1 \cdot x = 2
$$


单位矩阵$I$在矩阵方程中扮演着数字“1”的角色。我们解决$Ax = b$的目标是找到一种方法来分离向量 (vector)$x$。我们不能“除以”矩阵，但我们可以乘以一个称为**逆矩阵**的特殊矩阵（我们将在下一节讨论），以达到类似的效果。这个过程会是这样的：


$$
A^{-1} A x = A^{-1} b
$$


其中乘积$A^{-1} A$得到单位矩阵$I$：


$$
I x = A^{-1} b
$$


并且由于$Ix = x$，我们得到解：


$$
x = A^{-1} b
$$


单位矩阵是我们试图在方程一侧达到的目标，以找到我们的解。

### 在NumPy中创建单位矩阵

NumPy使创建单位矩阵变得简单。你可以使用`np.identity()`函数，它接受一个整数参数 (parameter)来指定大小。

```python
import numpy as np

# 创建一个3x3单位矩阵
I_3 = np.identity(3)
print("一个3x3单位矩阵:")
print(I_3)

# 创建一个5x5单位矩阵
I_5 = np.identity(5)
print("\n一个5x5单位矩阵:")
print(I_5)
```

让我们使用NumPy来验证乘法单位元性质。我们将创建一个矩阵`A`并将其乘以`I`。在NumPy中，`@`运算符用于矩阵乘法。

```python
# 定义一个3x2矩阵 A
A = np.array([
    [8, 1],
    [6, 3],
    [5, 7]
])

# 为 A @ I 创建一个2x2单位矩阵
I_2 = np.identity(2)

# 执行乘法运算
result = A @ I_2

print("矩阵 A:\n", A)
print("\n单位矩阵 I:\n", I_2)
print("\nA @ I 的结果:\n", result)

# 我们可以检查结果是否与原始矩阵 A 相同
are_equal = np.array_equal(A, result)
print(f"\n结果与 A 相同吗？ {are_equal}")
```

正如代码所示，将`A`乘以一个正确大小的单位矩阵会得到`A`本身。这个简单而有用的矩阵是你求解方程组时会经常用到的基本工具。

## 参考资料

- [18.06 Linear Algebra, Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Gilbert Strang (2010)
  Publisher: MIT OpenCourseWare
  这门由Gilbert Strang教授的开放课程提供了线性代数概念的解释，包括单位矩阵及其作用。
- [Deep Learning](http://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第二章介绍了线性代数，在机器学习的背景下解释了单位矩阵等概念。
- [numpy.identity](https://numpy.org/doc/stable/reference/generated/numpy.identity.html) — NumPy Developers (2024)
  Publisher: NumPy
  NumPy官方文档，介绍了如何使用`np.identity()`函数生成单位矩阵。
- [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) — NumPy Developers (2024)
  Publisher: NumPy Project
  NumPy官方文档，详细介绍了NumPy中的矩阵乘法，用于演示单位矩阵的乘法属性。
