---
course: "prompt-engineering-llm-application-development"
chapter: "building-applications-llm-frameworks"
lesson: "introduction-llm-frameworks-langchain"
sourceId: 3043
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/introduction-llm-frameworks-langchain"
title: "大型语言模型框架介绍（如LangChain）"
description: "理解使用大型语言模型应用程序开发框架的优势和核心理念。"
order: 1
plots: []
sourceHash: "e98cd6a00d849971754d3ee6c8a377fea85b98e3fd0e0614369b263a8e4da4fb"
sourceCorrections: []
---

尽管直接操作大型语言模型（LLM）API提供了最大的灵活性，但这种方式常常暴露出一些重复的模式。设置API调用、管理参数 (parameter)、解析响应，特别是串联多个大型语言模型交互，可能导致代码重复，并随着应用程序复杂度的提升而难以管理。从零开始构建时，处理对话历史或整合外部数据源会进一步增加难度。

大型语言模型应用程序开发框架正是在此发挥作用。可以将它们看作专门用于构建集成大型语言模型的软件的工具集。它们在原始API调用的基础上提供更高层级的抽象，提供预置的组件和结构，以简化开发流程。就像Web框架（如Flask、Django、Ruby on Rails或Express）通过处理路由、请求处理和模板来简化Web应用程序开发一样，大型语言模型框架提供了为大型语言模型核心任务量身定制的构建模块。

### 为什么使用大型语言模型框架？

使用框架有多个优势，尤其当你的应用程序不再是简单的单次提示时：

- **模块化与可复用性：** 框架鼓励将应用程序拆分为独立、可复用的组件。常见组件包括不同大型语言模型提供商的接口、用于动态生成提示的模板，以及从大型语言模型输出中提取结构化信息的解析器。这种模块化使代码更整洁，更易于维护。
- **组合性：** 它们提供了连接这些组件的标准化方式。一种常见模式是“链”，它按顺序连接组件，例如：接收用户输入、使用提示模板格式化、发送给大型语言模型，然后解析输出。框架使定义和执行这些链变得简单直接。
- **标准化：** 许多框架为与各种大型语言模型提供商（OpenAI、Anthropic、Cohere、开源模型等）交互提供统一接口。这使你能够以最小的代码改动在不同模型间切换，有助于试验和优化。
- **状态管理（记忆）：** 实现对话式应用程序需要管理交互历史。框架通常提供内置的“记忆”组件，自动处理过往消息的存储和获取，并将相关历史注入后续提示。
- **集成能力：** 它们简化了大型语言模型与其他资源的连接。这包括整合外部数据（如在检索增强生成 (RAG)RAG中所示）、使用外部API（如搜索引擎、天气服务或计算器），或与数据库交互。
- **代理抽象：** 对于需要推理 (inference)和工具使用的更复杂任务，框架通常提供构建“代理”的结构。这些代理使用大型语言模型作为推理引擎，根据用户输入决定采取哪些行动（例如，使用哪个工具）。
- **减少样板代码：** 通过处理API请求格式化、错误处理、重试和基本输出解析等常见任务，框架大幅减少了你所需编写的重复代码量。

### 介绍LangChain

在本章中，我们将主要使用 **LangChain** 作为示例框架。LangChain是目前最受欢迎、最全面的开源大型语言模型应用开发框架之一。其核心理念围绕组合性，允许开发者将各种组件串联起来构建复杂的应用程序。

我们将在后续部分分析LangChain的基本构建块：

- **模型：** 用于与不同类型语言模型（大型语言模型、聊天模型、嵌入 (embedding)模型）交互的标准化接口。
- **提示：** 使用模板进行动态提示构建和管理的工具。
- **解析器：** 用于从模型输出中提取结构化信息的实用工具。
- **链：** 将组件组合成序列以执行特定任务的核心理念。
- **记忆：** 用于在交互间保持应用程序状态的组件，对于聊天机器人而言不可或缺。
- **代理与工具：** 用于创建大型语言模型决定使用可用工具执行哪些行动的系统的抽象。

以下是这些组件在LangChain这类框架中可能如何交互的简化视图：

> 这张图表说明了大型语言模型框架中的组件可能如何交互。简单流程通常涉及将提示、模型和解析器串联起来，而更高级的代理流程则涉及决策和工具使用。

### 其他框架

尽管我们主要关注LangChain，但值得一提的是，也存在其他框架，例如LlamaIndex（通常侧重于RAG能力）和微软的Semantic Kernel。每个框架可能在设计理念、优势和抽象层级上略有不同。然而，模块化组件、组合和集成等基本思想在大多数高效的大型语言模型框架中都是共通的。理解其中一个为审视其他框架提供了坚实的基础。

### 权衡考量

大型语言模型框架在复杂应用的结构和开发速度方面提供了显著优势。然而，它们也引入了一层抽象。这意味着你可能需要花费一些初始时间学习框架的特定组件和约定，而不是直接操作API请求。对于非常简单的任务，使用框架可能显得大材小用。但随着应用程序复杂度的增加，模块化、可维护性和内置功能的优势通常会超过初始学习投入。对直接API交互（第4章）的良好理解对于调试和理解框架的内部运作原理仍然有价值。

在后续部分，我们将更详细地审视LangChain等框架提供的核心组件，从模型、提示和解析器开始。

## 参考资料

- [LangChain Documentation](https://www.langchain.com/docs/) — LangChain Developers (2024)
  Publisher: LangChain
  官方文档，提供框架的指南和API参考。
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex Developers (2024)
  Publisher: LlamaIndex
  构建数据增强型LLM应用的官方文档。
- [Applied LLMs: Your Guide to Building with Open-Source Large Language Models](https://www.packtpub.com/product/applied-llms-your-guide-to-building-with-open-source-large-language-models/9781803248677) — Itamar Regev, Lior Kandel (2023)
  Publisher: Packt Publishing
  一本关于LLM应用开发的书籍，内容涉及框架中使用的模式。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint arXiv:2302.04761; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  关于LLM学习使用外部工具的研究论文，LLM框架促进了这一能力。
