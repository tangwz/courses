---
course: "prompt-engineering-llm-application-development"
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag"
sourceId: 674
chapter: "integrating-llms-external-data-rag"
title: "整合大型语言模型与外部数据 (RAG)"
order: 6
description: "学习检索增强生成 (RAG) 方法，以连接大型语言模型与您自己的数据来源。"
hasQuiz: true
---

大型语言模型拥有广泛的通用知识，但这些知识通常在模型训练时就已固定。它们无法获取实时信息、公司内部文件或在训练结束后才产生的特定数据集。这种局限性限制了它们在需要当前或专有背景信息的应用中的有效性。

本章介绍检索增强生成 (RAG)，这是一种旨在解决此问题的方法。RAG 使大型语言模型能够动态地获取并使用外部信息。模型无需完全依赖其内部参数，而是可以将相关检索到的数据整合到生成过程中。

您将学到 RAG 系统的核心组成部分和工作流程。我们将介绍：

*   检索增强生成背后的理念以及为何需要它。
*   加载和准备用于检索的外部文档（拆分）的方法。
*   文本嵌入模型如何将文本转换为可搜索的向量表示。
*   向量存储在索引和查询这些嵌入中的作用。
*   实现检索过程以根据用户查询找到相关信息。
*   使用检索到的上下文来增强大型语言模型提示的方法。
*   逐步构建一个基本的 RAG 流程。

到本章结束时，您将了解如何将大型语言模型与外部知识来源连接起来，大幅扩展它们在特定文档问答或获取最新信息等任务中的适用性。
