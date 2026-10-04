---
course: "building-ml-recommendation-system"
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-2-content-based-filtering"
sourceId: 1293
chapter: "content-based-filtering"
title: "基于内容的过滤"
order: 2
description: "学习通过物品特征、针对文本的 TF-IDF 以及余弦相似度来构建基于内容的推荐系统。"
hasQuiz: false
---

在了解了推荐系统的基本构成后，我们现在将构建第一类推荐器：基于内容的过滤器。这种方法的逻辑十分直观，它根据物品的固有属性进行推荐。如果用户喜欢某部包含特定演员和类型的电影，基于内容的系统就会推荐具有相同特征的其他电影。它的运作原理很简单：如果你喜欢某个物品，你可能也会喜欢与之相似的另一个物品。

本章介绍构建此类系统的完整流程。我们首先将物品表示为特征向量，这一程序被称为创建物品画像。你将学习处理物品元数据，包括如何使用词频-逆文档频率 (TF-IDF) 技术将非结构化文本转换为数值格式。

将物品表示为向量后，下一步是衡量它们之间的相似度。我们将实现余弦相似度，这是处理此类任务的常用指标，用于计算两个向量 $A$ 和 $B$ 之间夹角的余弦值：

$$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$

最后，我们将整合这些组件，生成一个总结用户偏好的用户画像，并产出排序后的推荐列表。本章末尾设有一个动手实践环节，你将应用这些技术，从零开始构建一个可运行的电影推荐器。
