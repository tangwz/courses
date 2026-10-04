---
course: "getting-started-rag"
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-3-data-preparation-for-rag"
sourceId: 749
chapter: "data-preparation-for-rag"
title: "准备检索所需数据"
order: 3
description: "学习如何有效加载、处理和分块文档，以准备你的知识库供RAG系统使用。"
hasQuiz: true
---

前面讨论过的检索组件需要获取结构良好的信息才能有效运作。原始文档，无论是PDF、文本文件还是网页，都需要进行专门的准备，才能被高效地索引和搜索。本章侧重于将你的知识来源转化为适合检索增强生成（RAG）系统的实际步骤。

你将学习从不同来源摄取文档的方法，以及将大型文档拆分成更小、更易处理的片段（这一过程被称为分块）的原理。我们将介绍各种分块策略，从简单的固定大小拆分到更注重内容的技术。此外，你将了解如何将有意义的元数据与这些分块关联起来，以及将最终处理好的数据及其向量嵌入存储到向量数据库中，使其可供检索组件使用。实践练习将指导你使用常用库实现文档加载和分块。
