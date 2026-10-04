---
course: "getting-started-rag"
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-2-rag-retrieval-component"
sourceId: 748
chapter: "rag-retrieval-component"
title: "检索组件"
order: 2
description: "侧重于 RAG 中的检索器。了解向量嵌入、相似度搜索以及如何建立检索机制。"
hasQuiz: true
---

检索增强生成（RAG）系统能否提供准确和相关的回复，很大程度上取决于其第一阶段：检索到正确的信息。本章着重讲解这一检索机制。

你将学习文本如何被转换为数字表示，即向量嵌入，它们体现了语义信息。我们将研究相似度搜索技术，它们通常使用余弦相似度（$cos(\theta)$）等度量标准，来在一个大型集合中找出与用户查询最相关的嵌入。这自然就涉及到向量数据库——为存储和高效查询这些高维向量而构建的专用系统。我们还将介绍用于创建嵌入的常用模型，选择合适向量数据库时的考量，最后会有一个供你亲自生成文本嵌入的动手练习。
