---
course: "llm-alignment-safety"
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-4-evaluating-llm-safety-alignment"
sourceId: 690
chapter: "evaluating-llm-safety-alignment"
title: "评估LLM的安全性和对齐度"
order: 4
description: "涵盖评估大型语言模型安全性和对齐度的先进技术。"
hasQuiz: false
---

在考察了大型语言模型的对齐方法后，下一步自然是明确这些方法的效果如何，以及所得模型是否真正安全。构建LLM是过程的一部分；严格评估其行为是另一个重要环节。如果没有扎实的评估，关于对齐和安全性的说法就缺乏依据。

本章提供进行此项评估所需的技术和框架。我们将介绍：

*   定义并付诸实施主要安全维度，这些维度通常概括为有用性、诚实性和无害性（HHH）。
*   运用HELM和TruthfulQA等自动化基准测试来衡量特定能力和失效模式。
*   实行结构化人工评估规程和红队策略，以发现自动化测试遗漏的细微问题。
*   介绍用于量化模型输出中偏见和公平性的衡量指标和方法。
*   评估模型在分布变化下的表现，这是实际应用可靠性的一个重要考量。
*   讨论实现可扩展且持续可靠评估的实际难题。

学完本章，你将了解如何运用多方面方法来评估LLM的对齐性和安全特征，从而从零散的检查转向更系统的分析。
