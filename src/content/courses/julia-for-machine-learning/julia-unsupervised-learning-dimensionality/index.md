---
course: "julia-for-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality"
sourceId: 1208
chapter: "julia-unsupervised-learning-dimensionality"
title: "无监督学习与降维"
order: 4
description: "在 Julia 中应用 K-均值聚类和 PCA 等无监督学习技术进行降维。"
hasQuiz: false
---

在之前的章节中，我们的工作主要围绕监督学习展开。在这种学习中，算法从包含已知结果或标签的数据中进行学习。本章将重心转向无监督学习。机器学习的这个分支处理没有预定义标签的数据集。在此，目标是让算法自主找出数据中的模式、结构或关联。

我们还将讨论降维。具有许多特征的数据集可能在建模和计算方面带来挑战。降维技术旨在减少输入变量的数量，同时保留数据中有用的属性。这可以带来更简单的模型、更快的计算，并能通过将数据投影到低维空间来辅助数据可视化，例如从 $R^n$ 到 $R^2$ 或 $R^3$。

在本章中，您将学习如何在 Julia 中实现几种常用的无监督学习算法。我们将介绍聚类方法，例如 K-均值 (K-Means)，它将数据划分成指定数量 $k$ 的簇；以及 DBSCAN，一种基于密度的方法。此外，您还将使用主成分分析 (PCA)，这是一种常用的数据降维技术。最后，我们将审视评估聚类结果表现的方法。
