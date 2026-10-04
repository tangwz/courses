---
course: "python-llm-workflows"
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps"
sourceId: 967
chapter: "testing-evaluating-llm-apps"
title: "LLM 应用的测试与评估"
order: 9
description: "学习测试基于 Python 的 LLM 应用的可靠性和表现的策略和工具。"
hasQuiz: true
---

使用大型语言模型构建应用，在验证它们的行为和表现时，会遇到一些特别的难题。与传统软件的输出结果通常是确定的不同，LLM 的响应可能会变化，这使得标准测试方法本身不够用。评估 LLM 生成内容的*质量*和*可靠性*，需要一些特别的方法和考量。

本章讨论了测试和评估你已学习构建的基于 Python 的 LLM 应用的实际方面。我们将介绍：

*   在测试由 LLM 驱动的系统时遇到的特有难点。
*   单元测试特定组件的方法，例如提示模板和输出解析器。
*   集成测试整个 LLM 工作流程的策略。
*   评估应用表现的方法，包括相关指标和人工反馈的作用。
*   旨在协助 LLM 评估的框架介绍。
*   LLM 应用内部日志记录和监控的良好实践。

在本章结束时，你将明白如何实施结构化的测试流程和评估方法，这些方法是针对 LLM 驱动系统特点而设计的。
