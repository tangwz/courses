---
course: "llm-model-sizes-hardware"
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs"
sourceId: 837
chapter: "estimating-hardware-needs"
title: "估算硬件需求"
order: 5
description: "学习简单方法和经验法则，以估算运行特定LLM规模所需的显存。"
hasQuiz: true
---

在明确了LLM大小与所需硬件组成部分之间的联系后，本章将介绍估算这些需求的方法。您将学习如何估算运行特定大型语言模型所需的内存量，特别是显存（VRAM）。

我们将介绍一个常用的经验法则，它将模型参数与显存使用量关联起来，同时考虑$FP16$等数据类型。通常采用一个简化的计算公式：
$$Required \, VRAM \approx Parameter \, Count \times Bytes \, Per \, Parameter$$
除了这个初步估算，我们还将讨论影响内存需求的其他因素，例如处理过程中的激活内存占用、上下文长度以及批处理大小的影响。您还将学习如何检查自己系统的硬件规格，并通过实际例子应用这些估算技术。本章将为您提供实用的工具，以便在运行大型语言模型之前评估硬件需求。
