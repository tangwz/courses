# 第 2 章：矩阵：数据表示与变换

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-2-matrices-data-representation-transformations)

[返回课程目录](../README.md)

本章介绍矩阵，它将向量的思想扩展，用于处理数据集合。矩阵，本质上是一个数字网格，在机器学习中有两大主要作用：组织数据集和执行几何变换。

我们将讲解矩阵的基本运算，包括加法、减法、转置和乘法。你将学习向量乘以矩阵如何表示数据点上的线性变换，例如缩放和旋转。我们还将考察不同类型的矩阵（如单位矩阵和对角矩阵），并了解它们如何提供一种紧凑的方式来表示线性方程组，这些方程组通常写成 $Ax = b$ 的形式。

在本章中，我们将使用 Python 中的 NumPy 库高效地实现这些运算，为你将这些思想应用于实际的机器学习任务做准备。

## 小节

- 1. [矩阵用于组织数据](01-%E7%9F%A9%E9%98%B5%E7%94%A8%E4%BA%8E%E7%BB%84%E7%BB%87%E6%95%B0%E6%8D%AE.md)
- 2. [矩阵基本运算](02-%E7%9F%A9%E9%98%B5%E5%9F%BA%E6%9C%AC%E8%BF%90%E7%AE%97.md)
- 3. [矩阵乘法说明](03-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E8%AF%B4%E6%98%8E.md)
- 4. [矩阵作为线性变换](04-%E7%9F%A9%E9%98%B5%E4%BD%9C%E4%B8%BA%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2.md)
- 5. [常见矩阵类型](05-%E5%B8%B8%E8%A7%81%E7%9F%A9%E9%98%B5%E7%B1%BB%E5%9E%8B.md)
- 6. [表示线性方程组](06-%E8%A1%A8%E7%A4%BA%E7%BA%BF%E6%80%A7%E6%96%B9%E7%A8%8B%E7%BB%84.md)
- 7. [使用 NumPy 进行矩阵运算](07-%E4%BD%BF%E7%94%A8%20NumPy%20%E8%BF%9B%E8%A1%8C%E7%9F%A9%E9%98%B5%E8%BF%90%E7%AE%97.md)
- 8. [动手实践：数据点变换](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E7%82%B9%E5%8F%98%E6%8D%A2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-2-matrices-data-representation-transformations/quiz)
