---
course: "building-llm-apps-with-langchain"
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents"
sourceId: 1289
chapter: "developing-autonomous-agents"
title: "开发自主智能体"
order: 6
description: "学习创建自主智能体，它们使用大型语言模型进行推理、做出决策，并使用工具与外部系统互动。"
hasQuiz: false
---

之前的章节侧重于使用链（chains）构建具有预设顺序逻辑的应用。现在我们将转向一个更动态的模型，其中大型语言模型决定达成目标所需的行动顺序。这就是智能体（agent）的作用。智能体使用大型语言模型作为其推理引擎，通过一组可用的工具与环境进行互动。

本章介绍构建这些系统的组成部分。您将学习智能体、工具和工具包的架构。我们将讲解如何为智能体提供内置功能，例如网页搜索，以及用于与特定API交互的自定义工具。您还将研究不同类型的智能体以及运行推理循环的`AgentExecutor`。本章最后将构建一个实用智能体，能够查找并整合信息以回答复杂问题。
