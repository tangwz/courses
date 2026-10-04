---
course: "building-ml-recommendation-system"
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-3-neighborhood-based-collaborative-filtering"
sourceId: 1294
chapter: "neighborhood-based-collaborative-filtering"
title: "基于邻域的协同过滤"
order: 3
description: "使用 k-近邻和相似度指标实现基于用户和基于项目的协同过滤。"
hasQuiz: false
---

在上一章中，我们通过分析项目的固有属性生成了推荐。现在，我们将注意力从项目属性转向协同过滤中的用户行为。这项技术基于一个简单、直观的原理：过去意见一致的用户，在未来也可能达成一致。我们不再询问“这个项目是什么样的？”，而是询问“还有谁喜欢这个项目？”。

本章介绍基于邻域的方法，这是许多协同过滤系统的底层逻辑。我们将首先把数据整理成用户-项目交互矩阵，这是一个网格，其中每个单元格 $R_{u,i}$ 代表用户 $u$ 与项目 $i$ 之间的交互。

由此，你将学会：

*   区分基于用户和基于项目的协同过滤。
*   通过应用 k-近邻 (k-NN) 算法找到相似的用户或项目。
*   使用余弦相似度和皮尔逊相关系数等指标计算相似度得分。
*   根据“邻居”的评分来预测用户对某个项目的潜在评分。
*   处理交互矩阵中常见的数据稀疏问题。

学习完本章后，你将构建一个实用的基于项目的协同过滤器，让你实际掌握如何纯粹根据用户交互模式来生成推荐。
