# 第 6 章：机器学习中的矩阵分解

来源：[原章节](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning)

[返回课程目录](../README.md)

矩阵表示数据和变换，但它们的结构有时可能很复杂。矩阵分解技术提供了一种将矩阵分解成更简单的组成矩阵的方法。这种分解通常使理解矩阵的性质并有效执行某些计算变得更容易。

本章介绍机器学习场景中常用的一些重要矩阵分解方法。你将学到：

*   **奇异值分解 (SVD)**: 一种功能强大的分解 ($A = U\Sigma V^T$)，适用于任意 $m \times n$ 矩阵。我们将了解它的组成部分（奇异值和奇异向量）、它的几何意义，以及它在降维、降噪和数据压缩中的应用。
*   **LU 分解**: 将一个方阵分解成一个下三角矩阵 ($L$) 和一个上三角矩阵 ($U$) 的乘积，通常写作 $A = LU$。这对于求解线性方程组 $Ax=b$ 尤其有用。
*   **QR 分解**: 将矩阵分解成一个正交矩阵 ($Q$) 和一个上三角矩阵 ($R$)，使得 $A = QR$。这种方法在与最小二乘问题相关的算法中很常见。

我们将讨论 SVD 与上一章中介绍的特征分解之间的关系。还会介绍使用 NumPy 和 SciPy 等 Python 库实现这些分解的实际操作，使你能够将这些技术应用于数据。

## 小节

- 1. [矩阵分解简介](01-%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3%E7%AE%80%E4%BB%8B.md)
- 2. [奇异值分解 (SVD)](02-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3%20%28SVD%29.md)
- 3. [SVD 的几何含义](03-SVD%20%E7%9A%84%E5%87%A0%E4%BD%95%E5%90%AB%E4%B9%89.md)
- 4. [SVD在降维中的应用](04-SVD%E5%9C%A8%E9%99%8D%E7%BB%B4%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
- 5. [SVD在数据压缩中的应用](05-SVD%E5%9C%A8%E6%95%B0%E6%8D%AE%E5%8E%8B%E7%BC%A9%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
- 6. [奇异值分解与特征分解的关系](06-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3%E4%B8%8E%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3%E7%9A%84%E5%85%B3%E7%B3%BB.md)
- 7. [LU 分解概述](07-LU%20%E5%88%86%E8%A7%A3%E6%A6%82%E8%BF%B0.md)
- 8. [QR分解概览](08-QR%E5%88%86%E8%A7%A3%E6%A6%82%E8%A7%88.md)
- 9. [使用 SciPy/NumPy 实现分解](09-%E4%BD%BF%E7%94%A8%20SciPy-NumPy%20%E5%AE%9E%E7%8E%B0%E5%88%86%E8%A7%A3.md)
- 10. [SVD实践应用](10-SVD%E5%AE%9E%E8%B7%B5%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/quiz)
