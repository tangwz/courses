---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "introduction-to-langchain"
sourceId: 5135
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/introduction-to-langchain"
title: "LangChain 介绍"
description: "LangChain 的目的、架构和主要组成概述。"
order: 1
plots: []
sourceHash: "8a33205ec92cc699c74b65658faf3c9c010e2c94f3a617a4c2baa1d0c9a8e9c0"
sourceCorrections: []
---

直接使用 `requests` 等 Python 库或供应商特定的 SDK 与大型语言模型 (LLM) API 交互很重要。这让您可以发送提示并接收补全。然而，构建比简单问答更复杂的应用程序通常涉及多个步骤：动态格式化提示，可能需要对 LLM 进行多次调用，与外部工具（如搜索引擎或数据库）交互，以及构造最终输出。手动管理这种繁琐性会很快变得麻烦且容易出错。

这就是 LangChain 出现的地方。LangChain 是一个开源框架，旨在简化使用语言模型开发应用程序。它提供了一个标准、可扩展的接口和组件，用于创建精细的工作流程。可以将其视为一个工具包，帮助您组装 LLM 驱动应用程序所需的构成要素，而不是从零开始构建每个连接和交互。

使用 LangChain 这样框架的主要目的是管理复杂情况并提高模块化程度。LangChain 鼓励您将应用程序分解为独立、易于管理的部分，而不是编写庞大的脚本。它为常见任务提供了抽象，例如：

- **与模型交互：** 提供与各种 LLM 提供商（如 OpenAI、Anthropic、Cohere 或托管在 Hugging Face 上的开源模型）交互的一致方式，而无需学习每个具体 API 的细节。
- **管理提示：** 提供工具，用于创建动态、可重用的提示模板，这些模板可以包含用户输入、之前步骤的上下文 (context)或从外部来源获取的数据。
- **结构化输出：** 包含名为“输出解析器”的实用程序，有助于将 LLM 通常是非结构化的文本输出转换为更易用的格式，例如 JSON 对象或 Python 数据类。
- **连接组件：** 支持创建“链”（Chains），它定义操作序列，将提示、模型、解析器和其他工具连接起来，以执行更复杂的任务。我们将在下一章中更详细地介绍链。

本质上，LangChain 提供了一组可以组合的构成要素或模块。本章我们将重点关注的几个重要模块是 `Models`、`Prompts` 和 `Output Parsers`。

> LangChain 工作流程的简要视图：用户输入由提示模板格式化，发送给 LLM 模型，响应再由输出解析器结构化。

这种基于组件的方法使您的代码更简洁、更易于调试且易于调整。如果您想替换一个 LLM 或改变输出解析方式，通常只需要修改相应组件，而无需重写应用程序的大部分逻辑。

在以下部分中，我们将详细研究这些 LangChain 主要组件，首先介绍 LangChain 如何抽象化与不同语言模型的交互。您将学会如何使用这些构成要素，使用 LangChain 框架构建您的第一个简单 LLM 应用程序。

## 参考资料

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) — LangChain Community (2024)
  理解LangChain架构、模块和API的主要官方资源。
- [The Architecture of a New Generative AI Application](https://blog.langchain.dev/the-architecture-of-a-new-generative-ai-application/) — Harrison Chase (2023)
  Publisher: LangChain Blog
  阐释构建复杂LLM应用框架背后的设计原则和动机。
