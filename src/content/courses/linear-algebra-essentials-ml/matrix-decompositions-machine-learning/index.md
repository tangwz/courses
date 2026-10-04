---
course: "linear-algebra-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning"
sourceId: 471
chapter: "matrix-decompositions-machine-learning"
title: "机器学习中的矩阵分解"
order: 6
description: "了解 SVD、LU 和 QR 等重要的矩阵分解，理解它们的性质及其在机器学习中的应用。"
hasQuiz: true
---

矩阵表示数据和变换，但它们的结构有时可能很复杂。矩阵分解技术提供了一种将矩阵分解成更简单的组成矩阵的方法。这种分解通常使理解矩阵的性质并有效执行某些计算变得更容易。

本章介绍机器学习场景中常用的一些重要矩阵分解方法。你将学到：

*   **奇异值分解 (SVD)**: 一种功能强大的分解 ($A = U\Sigma V^T$)，适用于任意 $m \times n$ 矩阵。我们将了解它的组成部分（奇异值和奇异向量）、它的几何意义，以及它在降维、降噪和数据压缩中的应用。
*   **LU 分解**: 将一个方阵分解成一个下三角矩阵 ($L$) 和一个上三角矩阵 ($U$) 的乘积，通常写作 $A = LU$。这对于求解线性方程组 $Ax=b$ 尤其有用。
*   **QR 分解**: 将矩阵分解成一个正交矩阵 ($Q$) 和一个上三角矩阵 ($R$)，使得 $A = QR$。这种方法在与最小二乘问题相关的算法中很常见。

我们将讨论 SVD 与上一章中介绍的特征分解之间的关系。还会介绍使用 NumPy 和 SciPy 等 Python 库实现这些分解的实际操作，使你能够将这些技术应用于数据。
