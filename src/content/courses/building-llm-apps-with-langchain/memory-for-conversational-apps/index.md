---
course: "building-llm-apps-with-langchain"
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-4-memory-for-conversational-apps"
sourceId: 1287
chapter: "memory-for-conversational-apps"
title: "会话应用中的记忆功能"
order: 4
description: "学会如何为你的LLM应用添加状态和记忆，以构建像聊天机器人和助手这样的会话界面。"
hasQuiz: false
---

大型语言模型本质上是无状态的。每个API调用都是独立处理的，没有内置的对之前交互的记忆。这给构建聊天机器人或助手等应用带来了直接的难题，因为保持会话的上下文对于连贯且有用的用户体验来说是必需的。

本章将介绍LangChain的记忆组件，这些组件旨在解决此问题。你将学会如何为你的链和代理添加状态，使它们能够在单个会话中记住并引用过去的交流。

我们将讲解多种记忆管理策略，每种都适用于不同的使用场景。你将从像`ConversationBufferMemory`这样的简单方法开始，它保留了完整的聊天历史记录。对于令牌限制成为考量因素的较长会话，我们将审视`ConversationSummaryMemory`，它使用大型语言模型来浓缩历史记录。我们还将讲解管理固定大小上下文的技巧，例如窗口式和基于令牌的记忆类型。

最后，你将看到将这些记忆对象集成到你的应用中的实际操作步骤。本章以一个实践练习结束，该练习旨在构建一个能够从一轮到下一轮保持会话上下文的聊天机器人。
