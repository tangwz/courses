---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents"
sourceId: 1352
chapter: "developing-autonomous-agents"
title: "构建自主代理"
order: 8
description: "学习如何借助 `agent` 模块构建能进行推理、规划并使用工具来解决问题的自主代理。"
hasQuiz: false
---

到目前为止，我们已将大型语言模型（LLMs）用于直接的、单轮任务，例如生成文本或依据给定背景信息回答问题。本章将讲解构建能够更自主运作以达成目的的系统。自主代理使用大型语言模型（LLM）作为推理引擎来确定一系列步骤，并通常会与外部工具交互以收集信息或执行操作。此过程可表示为思维、行动和观察的循环。

在本章中，您将学习使用 `agent` 模块来构建这些代理。我们将从介绍 ReAct（推理与行动）模式开始，它是一个用于组织代理行为的常用架构。接着，您将实现一个 ReAct 代理，并学习如何为其提供工具，例如搜索功能或一个简单的计算器，以扩展其功能。核心循环可简化为以下步骤：

$$
\text{Goal} \rightarrow \text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation} \rightarrow \dots
$$

接下来，我们将讲解另一种架构，名为“规划-执行”代理。最后，我们将讨论允许多个代理合作以处理更复杂问题的系统构建原则。本章结束时，您将能够构建能将问题分解、使用工具寻找处理办法并代表您行事的代理。
