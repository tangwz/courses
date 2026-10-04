# 第 5 章：特征值与特征向量

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors)

[返回课程目录](../README.md)

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

## 小节

- 1. [矩阵作为线性变换](01-%E7%9F%A9%E9%98%B5%E4%BD%9C%E4%B8%BA%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2.md)
- 2. [定义特征值和特征向量](02-%E5%AE%9A%E4%B9%89%E7%89%B9%E5%BE%81%E5%80%BC%E5%92%8C%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F.md)
- 3. [几何解释](03-%E5%87%A0%E4%BD%95%E8%A7%A3%E9%87%8A.md)
- 4. [特征方程](04-%E7%89%B9%E5%BE%81%E6%96%B9%E7%A8%8B.md)
- 5. [动手实践：使用NumPy求解特征值](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8NumPy%E6%B1%82%E8%A7%A3%E7%89%B9%E5%BE%81%E5%80%BC.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors/quiz)
