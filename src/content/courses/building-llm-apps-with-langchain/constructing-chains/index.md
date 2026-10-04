---
course: "building-llm-apps-with-langchain"
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-3-constructing-chains"
sourceId: 1286
chapter: "constructing-chains"
title: "构建用于顺序操作的链"
order: 3
description: "学习连接大型语言模型（LLM）及其他组件，以使用 LangChain 中不同类型的链来创建复杂的、多步骤的应用。"
hasQuiz: false
---

上一章，我们组装了与语言模型交互的基本组成部分：提示、模型和输出解析器。这些组件使得一次性、结构化地调用大型语言模型（LLM）成为可能。然而，大多数应用需要多个步骤。例如，您可能需要先生成一个主题摘要，然后使用该摘要来撰写一篇文章。这就需要将操作串联起来。

LangChain 的“链”正是为此目的而设计。它们提供一个标准化的接口，用于将各个组件连接成逻辑顺序。单次LLM调用可以看作是一个函数 $f$，它接收一个输入并产生一个输出。
$$
\text{Output} = f(\text{Input})
$$
链允许您组合这些函数，从而创建多步骤的序列，其中一个步骤的输出成为下一个步骤的输入，例如 $g(f(\text{Input}))$。

本章介绍如何构建这些序列。您将学习：
*   使用基本的 `LLMChain` 来连接模型、提示和解析器。
*   使用 `SequentialChain` 构建多步骤工作流，以按固定顺序执行操作。
*   借助 `RouterChain`，将条件逻辑引入工作流，以根据输入选择不同的路径。
*   应用这些结构来构建内容生成流程，作为一项实践练习。
