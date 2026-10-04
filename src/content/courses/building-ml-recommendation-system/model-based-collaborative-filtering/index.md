---
course: "building-ml-recommendation-system"
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering"
sourceId: 1295
chapter: "model-based-collaborative-filtering"
title: "基于矩阵分解的模型协同过滤"
order: 4
description: "学习使用 SVD 等矩阵分解技术，通过用户-物品数据中的潜在因素来实现基于模型的协同过滤。"
hasQuiz: false
---

在上一章中，我们通过直接从交互数据中寻找用户或物品之间的相似性来构建推荐系统。尽管这些基于邻域的方法有效，但它们并不总是能捕捉到用户偏好背后的隐含模式。本章介绍基于模型的协同过滤，这是一种通过寻找潜在因素来解释用户与物品交互的不同方法。

我们要使用的主要技术是矩阵分解。其核心思路是将大型稀疏的用户-物品交互矩阵 $R$ 分解为两个较小的稠密矩阵，以此来进行近似处理。其中一个矩阵用潜在因素 ($P$) 表示用户，另一个矩阵则用同样的因素 ($Q$) 表示物品。接着，我们可以通过对应的用户向量和物品向量的点积，来预估原始矩阵中的任何评分：

$$
\hat{r}_{ui} = p_u \cdot q_i = \sum_{k=1}^{K} p_{uk}q_{ki}
$$

该模型有助于在观测到的评分之外进行泛化，并能为任何用户-物品组合提供推荐。

在本章中，你将学习如何从零开始实现这种方法。我们将从奇异值分解 (SVD) 开始，这是一种常用的矩阵分解算法。你将看到如何使用随机梯度下降 (SGD) 等优化技术来训练这些模型，以及如何应用正则化来防止过拟合。最后，你将通过一个完整的动手练习示例，使用流行库来构建并生成推荐。
