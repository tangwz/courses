---
course: "intro-llm-red-teaming"
sourceUrl: "https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-2-understanding-llm-attack-surfaces"
sourceId: 1133
chapter: "understanding-llm-attack-surfaces"
title: "了解 LLM 攻击面"
order: 2
description: "识别并理解大型语言模型中的各种攻击面，从提示注入到数据投毒。"
hasQuiz: true
---

为了有效保护大型语言模型，我们首先需要清楚地认识它们可能被攻击的方式。本章着重于识别和分析LLM及其周边系统中存在的各种攻击面。攻击面是指未经授权的用户（即攻击者）能够尝试向环境中输入数据或从中获取数据，或者以非预期的方式与系统进行交互的任何一个地方。

你将学习几种常见的攻击途径。我们将介绍诸如提示注入的技术，即在用户输入中嵌入恶意指令以操控模型的行为。我们还将讨论数据投毒，这涉及破坏训练数据以引入安全弱点或偏见。其他内容包括模型规避、绕过安全控制的越狱方法，以及攻击者可能如何尝试获取敏感信息。我们还将考虑诸如拒绝服务、虚假信息生成以及LLM API中的安全弱点等问题。在本章结束时，你将能够识别这些主要的攻击面，并理解它们对LLM安全可能造成的影响。
