---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-7-conversational-applications-with-memory"
sourceId: 1351
chapter: "conversational-applications-with-memory"
title: "构建具备记忆的对话式应用"
order: 7
description: "学习通过实现不同类型的记忆功能，来构建有状态的聊天机器人和对话式应用。"
hasQuiz: false
---

到目前为止，我们与语言模型的交流都是一问一答式的。我们发送请求，收到响应，而模型不记得之前的交流内容。这种无状态的特性在构建需要连续对话的应用（例如聊天机器人或助手）时，构成了一个明显的难题。如果没有保留上下文的机制，一次对话就只是一系列不连贯的问答。

本章将介绍记忆这种方法，以构建有状态的对话式应用。我们将使用 `memory` 模块来管理和保存对话历史，从而让你的应用在多轮对话中保持上下文。

你将学习如何实现几种记忆方案：

*   **对话缓冲记忆：** 一种直接的方式，它会逐字存储对话最近的来回内容。
*   **摘要记忆：** 一种做法，它借助大型语言模型逐步压缩对话历史，有助于管理长时间交流中的token数量。
*   **实体记忆：** 一种专门的方式，用于追踪对话中提及的特定信息片段，例如姓名或地点。
