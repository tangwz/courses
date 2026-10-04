---
course: "linear-algebra-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors"
sourceId: 469
chapter: "eigenvalues-eigenvectors"
title: "特征值与特征向量"
order: 5
description: "了解特征值与特征向量，它们的计算方法，几何意义，以及在降维（PCA）中的作用。"
hasQuiz: true
---

基于对矩阵作为线性变换的认识，本章将讨论一类特殊的向量，它们的方向在变换作用下保持不变，仅发生大小上的缩放。这些向量被称为特征向量，而缩放因子则是对应的特征值。

你将首先学习由方程 $Ax = \lambda x$ 所描述的严格定义。接着，我们将阐述这个方程背后的几何意义，直观地了解特征向量如何表示变换的“轴”。你将学习计算的标准方法：通过求解特征方程 $$det(A - \lambda I) = 0$$ 来找到特征值，并随后确定对应的特征向量。

我们将介绍特征分解，即将特定矩阵表示为其特征值和特征向量的过程，通常表示为 $A = PDP^{-1}$。机器学习中一个重要应用是主成分分析（PCA），它使用特征向量进行降维；我们将研究这种联系。最后，你将看到如何使用 Python 的 NumPy 库高效地完成这些计算。
