# 第 1 章：向量嵌入与向量空间

来源：[原章节](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-1-embeddings-vector-spaces-foundation)

[返回课程目录](../README.md)

为了构建能够理解含义的系统，我们首先需要一种将文本或图像等数据以数值形式表示的方法。本章主要讲解向量嵌入。向量嵌入作为这种数值表示方法，将数据点放置在高维向量空间中。

我们将从回顾不同类型数据如何转换为向量开始。您将学习常见的嵌入模型，特别是基于Transformer架构的模型，并讨论向量维度对系统性能的影响。我们还将介绍降维技术。向量处理的一个重要方面是衡量它们的相似性；因此，我们将比较余弦相似度 ($cos(\theta)$)、欧几里得距离 ($||\vec{a} - \vec{b}||_2$) 和点积 ($\vec{a} \cdot \vec{b}$) 等度量指标。最后，您将通过使用Python库生成嵌入并计算它们的相似性来应用这些知识。

## 小节

- 1. [从数据到向量：回顾](01-%E4%BB%8E%E6%95%B0%E6%8D%AE%E5%88%B0%E5%90%91%E9%87%8F%EF%BC%9A%E5%9B%9E%E9%A1%BE.md)
- 2. [嵌入模型概述](02-%E5%B5%8C%E5%85%A5%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
- 3. [理解向量维度](03-%E7%90%86%E8%A7%A3%E5%90%91%E9%87%8F%E7%BB%B4%E5%BA%A6.md)
- 4. [降维技术概述](04-%E9%99%8D%E7%BB%B4%E6%8A%80%E6%9C%AF%E6%A6%82%E8%BF%B0.md)
- 5. [测量向量空间中的相似度](05-%E6%B5%8B%E9%87%8F%E5%90%91%E9%87%8F%E7%A9%BA%E9%97%B4%E4%B8%AD%E7%9A%84%E7%9B%B8%E4%BC%BC%E5%BA%A6.md)
- 6. [动手实践：生成与比较嵌入](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%94%9F%E6%88%90%E4%B8%8E%E6%AF%94%E8%BE%83%E5%B5%8C%E5%85%A5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-1-embeddings-vector-spaces-foundation/quiz)
