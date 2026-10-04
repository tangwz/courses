---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-6-building-rag-systems"
sourceId: 1350
chapter: "building-rag-systems"
title: "构建检索增强生成 (RAG) 系统"
order: 6
description: "结合文档处理、嵌入和生成，构建一个完整的用于问答的 RAG 系统。"
hasQuiz: false
---

前面几章介绍了处理文档以及使用嵌入来查找语义相似文本的方法。本章将这些组件组合起来，以构建一个完整的问答系统。标准的大型语言模型 (LLM) 仅基于其预训练数据生成回复。检索增强生成 (RAG) 通过在生成时向模型提供外部的、最新的信息，从而扩展了此功能。

你将首先了解 RAG 系统的完整架构，从数据摄取到最终回复生成。接着，你将使用 `retrieval` 模块搭建你的第一个端到端流程。本章还会讲解提高检索准确性的方法，例如实现不同的搜索策略和应用重排序模型。最后，你将学习如何在将检索到的上下文传递给生成模型之前，对其进行正确格式化和管理。

在本章结束时，你将实际掌握如何构建和优化 RAG 系统，用于回答你文档中的问题。
