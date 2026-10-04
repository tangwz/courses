---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search"
sourceId: 1349
chapter: "embeddings-and-semantic-search"
title: "嵌入与语义搜索"
order: 5
description: "学习如何创建文本嵌入并进行语义相似度搜索以实现信息检索。"
hasQuiz: false
---

在上一章中，你学习了如何加载并将文档分割成易于处理的小块。为了使这些小块在检索时发挥作用，计算机必须能够根据其语义含义进行比较，而不仅仅是关键词。本章侧重于文本嵌入，它是一种捕捉文本含义的数值表示。通过将文本转换为向量，我们可以通过数学计算来为用户的查询找到最相关的信息。

本章会教你如何使用 Kerb 的 `embedding` 模块生成这些向量表示。你会学到向量相似度度量，例如余弦相似度，它用于衡量两段文本的关联程度。针对两个向量 $A$ 和 $B$ 的余弦相似度计算公式如下：

$$
\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$

基于此，你将构建一个语义搜索功能来找到相关的文档小块。本章最后会提供指导，说明如何根据性能、大小和你的特定应用场景等因素来选择适合的嵌入模型。
