---
course: "linear-algebra-fundamentals-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations"
sourceId: 1360
chapter: "systems-of-linear-equations"
title: "线性方程组"
order: 4
description: "学习如何使用矩阵表示法、逆和计算方法表示并求解线性方程组。"
hasQuiz: true
---

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
