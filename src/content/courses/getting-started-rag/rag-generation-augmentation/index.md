---
course: "getting-started-rag"
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation"
sourceId: 750
chapter: "rag-generation-augmentation"
title: "生成模块与增强"
order: 4
description: "学习检索到的信息如何用于增强提示词，并引导LLM在RAG系统中生成响应。"
hasQuiz: true
---

在之前的章节中，我们已经讲解了如何检索相关信息，现在我们将重点放在这些信息如何被加以运用。本章将审视检索增强生成（RAG）过程中的“生成”部分，具体说明大型语言模型（LLM）如何使用检索到的上下文来形成最终答案。

我们将介绍以下几个主要方面：

*   LLM模块在RAG系统中的作用和功能。
*   构建能有效整合原始查询和检索到的上下文的提示词方法。
*   将找到的文本段落注入LLM输入中的方法。
*   在添加检索到的信息时，管理LLM上下文窗口限制的策略。
*   LLM通过合成增强后的提示词来产生连贯响应的过程。
*   使用相关元数据将生成的输出追溯回源文档的方法。

完成本章后，您将清楚地了解检索到的数据如何积极地塑造和改进RAG管道中LLM的最终输出。
