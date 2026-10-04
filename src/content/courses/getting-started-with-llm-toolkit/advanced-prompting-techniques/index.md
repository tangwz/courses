---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-2-advanced-prompting-techniques"
sourceId: 1346
chapter: "advanced-prompting-techniques"
title: "高级提示技巧"
order: 2
description: "通过模板和少样本示例掌握提示工程。学习从大语言模型（LLM）输出中解析JSON等结构化数据。"
hasQuiz: false
---

在上一章中，你学会了进行一次基础的生成调用。然而，大语言模型（LLM）响应的质量直接取决于你所提供输入的质量。本章侧重于构建有效输入的技巧，这是一种被称为提示工程的做法。

我们将首先使用Kerb的模板引擎来构建可以通过变量动态填充的提示。然后你将了解如何组织和版本化这些提示，以便构建可维护的应用。为了提升模型在特定任务上的表现，我们将实践少样本提示（few-shot prompting），这包括在提示本身中提供示例。

一个普遍的需求是从模型中获取结构化数据，而不仅仅是自由格式的文本。我们将介绍指导LLM生成格式化输出的方法，然后使用 `parsing` 模块从模型的响应中准确提取结构化信息，例如JSON对象和代码块。这些技巧让你对模型的行为及其输出的格式有更精确的控制。
