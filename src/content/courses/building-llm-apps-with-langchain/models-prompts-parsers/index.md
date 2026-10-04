---
course: "building-llm-apps-with-langchain"
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers"
sourceId: 1285
chapter: "models-prompts-parsers"
title: "模型、提示与解析器"
order: 2
description: "了解如何与LLM通信，使用模板生成灵活的提示，并用LangChain的解析器整理模型输出。"
hasQuiz: false
---

任何LangChain应用的中心在于您的代码与大型语言模型（LLM）之间的通信。本章介绍处理这种通信的三个主要部分：模型、提示与解析器。

您将首先学习如何连接不同类型的模型，特别是LLM和聊天模型，并知道何时使用它们。之后，我们将说明如何使用`PromptTemplates`为这些模型构建准确且灵活的指令。您还将看到通过在提示中提供示例来改善模型响应的方法，这种方法称为少样本提示。最后，我们将介绍输出解析器，它们对于将模型自由格式的文本响应转换为有组织且可用的格式（例如JSON）是必需的。

这些部分构成了一个标准的调用顺序，通常表示为$Prompt \rightarrow Model \rightarrow Parser$。本章以一个实用的练习结束，您将在其中组合这些部分以制作一个从文本块中获取有组织数据的程序。
