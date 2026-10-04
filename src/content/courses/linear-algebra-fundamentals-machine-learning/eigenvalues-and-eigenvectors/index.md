---
course: "linear-algebra-fundamentals-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors"
sourceId: 1361
chapter: "eigenvalues-and-eigenvectors"
title: "特征值与特征向量"
order: 5
description: "对特征值和特征向量的介绍，以及它们在理解线性变换和降维方面所起的作用。"
hasQuiz: true
---

到目前为止，我们一直将矩阵视为用于乘法等运算的数字排列。现在，我们将视角转变为把矩阵看作执行线性变换的算子。当矩阵乘以向量时，它可以对向量进行拉伸、收缩或旋转，将其映射到空间中的一个新位置。

这引出了一个问题：对于一个给定的变换，是否存在方向不变的向量？答案存在于对特征值和特征向量的研究中。特征向量是指在变换中仅被缩放，而方向不发生变化的向量。对应的特征值是该缩放的标量因子。这种关系可以用以下方程简洁表示：

$$Av = \lambda v$$

其中，$A$ 是变换矩阵，$v$ 是特征向量，$\lambda$ (lambda) 是其对应的特征值。

本章中，我们将介绍以下主题：
*   如何将矩阵视为执行线性变换的函数。
*   特征值和特征向量的形式化定义。
*   这种向量与标量配对背后的几何意义。
*   使用特征方程计算特征值的方法。
*   如何使用 NumPy 计算给定矩阵的特征值和特征向量。
