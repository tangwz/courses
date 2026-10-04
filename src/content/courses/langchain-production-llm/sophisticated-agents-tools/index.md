---
course: "langchain-production-llm"
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools"
sourceId: 782
chapter: "sophisticated-agents-tools"
title: "构建精巧的代理与工具"
order: 2
description: "设计、实现和管理搭配自定义工具的精巧 LangChain 代理，以解决复杂问题。"
hasQuiz: false
---

在 LangChain 中建立了核心执行模型和定制化能力后，我们现在将注意力转向构建更自治的系统：**代理**。与预定义链不同，代理使用大型语言模型（LLM）作为推理引擎来决定行动序列，并常常通过**工具**与外部环境进行互动。

本章将指导您构建和管理这些精巧的代理。您将学到：

*   比较并实现不同的代理架构，例如 ReAct、Self-Ask 和 Plan-and-Execute。
*   开发自定义工具，赋予代理特定能力，例如与 API 或数据库进行互动。
*   实施处理错误的策略，这些错误可能在代理尝试使用工具时发生。
*   设计包含多个协作代理的系统。
*   使用较新 LLM 中提供的结构化工具调用（函数调用）功能，以实现更可靠的代理与工具互动。
*   追踪并分析代理的决策过程，以了解其行为并调试问题。

到本章结束时，您将能够设计出能够应对复杂、多步骤问题的代理，这些问题需要动态规划并与外部环境互动。
