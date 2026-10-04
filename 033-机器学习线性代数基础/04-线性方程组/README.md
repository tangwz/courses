# 第 4 章：线性方程组

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations)

[返回课程目录](../README.md)

在前几章中，我们明确了什么是向量和矩阵，以及如何对它们进行运算。现在，我们将运用这些工具来解决计算中最常见的问题之一：线性方程组。机器学习中的许多任务，例如将模型拟合到数据，都可以表述为一个需要求解的方程组。

本章将介绍如何以紧凑的矩阵形式表示一个包含多个方程和多个未知数的系统：

$$
Ax = b
$$

其中，$A$ 是一个已知系数的矩阵，$x$ 是我们希望找到的未知变量向量，$b$ 是已知结果的向量。我们的目标是找到解向量 $x$。

为此，你将学到：

*   矩阵逆及其在分离未知向量 $x$ 中的作用。
*   如何计算矩阵的行列式，并用它来检查是否存在唯一解。
*   可逆（非奇异）矩阵和不可逆（奇异）矩阵之间的区别。
*   使用 NumPy 的线性代数库求解方程 $Ax = b$ 的方法。

## 小节

- 1. [矩阵形式表示方程 (Ax = b)](01-%E7%9F%A9%E9%98%B5%E5%BD%A2%E5%BC%8F%E8%A1%A8%E7%A4%BA%E6%96%B9%E7%A8%8B%20%28Ax%20%3D%20b%29.md)
- 2. [单位矩阵](02-%E5%8D%95%E4%BD%8D%E7%9F%A9%E9%98%B5.md)
- 3. [矩阵的逆](03-%E7%9F%A9%E9%98%B5%E7%9A%84%E9%80%86.md)
- 4. [行列式与可逆性](04-%E8%A1%8C%E5%88%97%E5%BC%8F%E4%B8%8E%E5%8F%AF%E9%80%86%E6%80%A7.md)
- 5. [奇异矩阵与非奇异矩阵](05-%E5%A5%87%E5%BC%82%E7%9F%A9%E9%98%B5%E4%B8%8E%E9%9D%9E%E5%A5%87%E5%BC%82%E7%9F%A9%E9%98%B5.md)
- 6. [动手实践：使用NumPy求解方程组](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8NumPy%E6%B1%82%E8%A7%A3%E6%96%B9%E7%A8%8B%E7%BB%84.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/quiz)
