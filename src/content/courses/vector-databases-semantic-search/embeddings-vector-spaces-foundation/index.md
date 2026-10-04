---
course: "vector-databases-semantic-search"
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-1-embeddings-vector-spaces-foundation"
sourceId: 730
chapter: "embeddings-vector-spaces-foundation"
title: "向量嵌入与向量空间"
order: 1
description: "回顾向量嵌入（文本、图像）、嵌入模型（Transformer）、维度和相似度度量。动手生成嵌入。"
hasQuiz: true
---

为了构建能够理解含义的系统，我们首先需要一种将文本或图像等数据以数值形式表示的方法。本章主要讲解向量嵌入。向量嵌入作为这种数值表示方法，将数据点放置在高维向量空间中。

我们将从回顾不同类型数据如何转换为向量开始。您将学习常见的嵌入模型，特别是基于Transformer架构的模型，并讨论向量维度对系统性能的影响。我们还将介绍降维技术。向量处理的一个重要方面是衡量它们的相似性；因此，我们将比较余弦相似度 ($cos(\theta)$)、欧几里得距离 ($||\vec{a} - \vec{b}||_2$) 和点积 ($\vec{a} \cdot \vec{b}$) 等度量指标。最后，您将通过使用Python库生成嵌入并计算它们的相似性来应用这些知识。
