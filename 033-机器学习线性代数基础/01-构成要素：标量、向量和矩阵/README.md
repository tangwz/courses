# 第 1 章：构成要素：标量、向量和矩阵

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction)

[返回课程目录](../README.md)

本章介绍线性代数的基本对象。在进行复杂运算或构建机器学习模型之前，我们必须先了解如何用数学方式表示数据。为此，主要工具有标量、向量和矩阵。

我们将首先定义这三个组成部分。你将了解到，标量是一个单独的数，向量是表示空间中一个点的有序数列，而矩阵是用于组织整个数据集的数字网格。例如，一个包含多个特征的单个数据样本通常表示为一个向量，比如 $x = [x_1, x_2, x_3]$。这些样本的整个集合构成一个数据矩阵。

在此基础之上，我们将通过搭建一个包含NumPy库的Python环境来为接下来的实际操作做准备，NumPy是数值计算的标准工具。本章最后会有一个动手练习，你将应用所学知识在代码中创建你的第一个向量和矩阵。到本章结束时，你将能够使用线性代数的核心结构表示简单的数据集。

## 小节

- 1. [线性代数对机器学习有何作用？](01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%E5%AF%B9%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%9C%89%E4%BD%95%E4%BD%9C%E7%94%A8%EF%BC%9F.md)
- 2. [标量：最简单的元素](02-%E6%A0%87%E9%87%8F%EF%BC%9A%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84%E5%85%83%E7%B4%A0.md)
- 3. [向量：空间中的点](03-%E5%90%91%E9%87%8F%EF%BC%9A%E7%A9%BA%E9%97%B4%E4%B8%AD%E7%9A%84%E7%82%B9.md)
- 4. [矩阵：以网格形式组织数据](04-%E7%9F%A9%E9%98%B5%EF%BC%9A%E4%BB%A5%E7%BD%91%E6%A0%BC%E5%BD%A2%E5%BC%8F%E7%BB%84%E7%BB%87%E6%95%B0%E6%8D%AE.md)
- 5. [设置您的Python环境](05-%E8%AE%BE%E7%BD%AE%E6%82%A8%E7%9A%84Python%E7%8E%AF%E5%A2%83.md)
- 6. [动手实践：使用 NumPy 创建向量和矩阵](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%88%9B%E5%BB%BA%E5%90%91%E9%87%8F%E5%92%8C%E7%9F%A9%E9%98%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/quiz)
