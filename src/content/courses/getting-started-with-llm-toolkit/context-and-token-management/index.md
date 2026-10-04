---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-3-context-and-token-management"
sourceId: 1347
chapter: "context-and-token-management"
title: "管理上下文和令牌"
order: 3
description: "学习如何管理LLM上下文窗口，精确计算令牌，并使用tokenizer模块运用文本截断。"
hasQuiz: false
---

写出好用的提示后，下一个实际步骤是管理它们的大小。大型语言模型没有无限的记忆；它们在一个固定大小的输入范围内工作，这个范围被称为“上下文窗口”。发送超出此限制的提示会导致错误，而低效地使用可用空间会增加延迟和运行成本。API费用与令牌数量直接相关，通常遵循以下模型：$$Cost = (Price_{input} \times Tokens_{input}) + (Price_{output} \times Tokens_{output})$$。

本章着重介绍管理这一基本限制的工具和方法。您将学习：

*   使用 `tokenizer` 模块精确计算一段文本的令牌数量。
*   运用不同的文本截断策略，以确保提示适应模型的上下文窗口。
*   管理结合了多方信息的复杂提示的“令牌预算”。

本章结束后，您将能够控制您的应用程序的令牌使用，从而实现更可靠、更经济的性能。
