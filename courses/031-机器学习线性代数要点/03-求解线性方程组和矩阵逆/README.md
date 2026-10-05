# 第 3 章：求解线性方程组和矩阵逆

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses)

[返回课程目录](../README.md)

基于我们对向量和矩阵的理解，本章讨论求解线性方程组的基本问题。将这些方程组以矩阵形式表示为 $$Ax = b$$ 在机器学习中很常见，尤其是在确定线性回归等模型参数时。

在此，我们将考察寻找解向量 $x$ 的方法。主要议题包括：

*   矩阵逆 ($A^{-1}$) 的思想及其计算。
*   使用逆矩阵求解 $Ax = b$ 形式的方程组。
*   行列式在判断矩阵是否可逆中的作用。
*   计算方法，包括使用 NumPy 寻找逆矩阵和求解方程组，以及数值稳定性的考量。

我们还将简要回顾高斯消元法，作为求解这些方程组的一种基本方法。本章结束时，您将明白如何处理和求解在机器学习背景下频繁出现的线性方程组。

## 小节

- 1. [机器学习模型中的线性系统](01-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E7%BA%BF%E6%80%A7%E7%B3%BB%E7%BB%9F.md)
- 2. [高斯消元法简介](02-%E9%AB%98%E6%96%AF%E6%B6%88%E5%85%83%E6%B3%95%E7%AE%80%E4%BB%8B.md)
- 3. [矩阵逆](03-%E7%9F%A9%E9%98%B5%E9%80%86.md)
- 4. [计算矩阵的逆](04-%E8%AE%A1%E7%AE%97%E7%9F%A9%E9%98%B5%E7%9A%84%E9%80%86.md)
- 5. [行列式与可逆性](05-%E8%A1%8C%E5%88%97%E5%BC%8F%E4%B8%8E%E5%8F%AF%E9%80%86%E6%80%A7.md)
- 6. [运用逆矩阵求解 Ax=b](06-%E8%BF%90%E7%94%A8%E9%80%86%E7%9F%A9%E9%98%B5%E6%B1%82%E8%A7%A3%20Ax%3Db.md)
- 7. [数值稳定性和替代方法](07-%E6%95%B0%E5%80%BC%E7%A8%B3%E5%AE%9A%E6%80%A7%E5%92%8C%E6%9B%BF%E4%BB%A3%E6%96%B9%E6%B3%95.md)
- 8. [动手实践：求解模型系数](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B1%82%E8%A7%A3%E6%A8%A1%E5%9E%8B%E7%B3%BB%E6%95%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses/quiz)
