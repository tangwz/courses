---
course: "lora-peft-efficient-llm-training"
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-3-peft-methodologies-survey"
sourceId: 818
chapter: "peft-methodologies-survey"
title: "PEFT 方法概览"
order: 3
description: "了解其他主要的 PEFT 技术，例如适配器微调、前缀微调和提示微调。"
hasQuiz: false
---

在确立了低秩适配 (LoRA) 的原理与实现后，我们将视野转向其他主要的参数高效微调 (PEFT) 技术。本章将比较介绍为应对大型语言模型 (LLM) 微调计算需求而发展的不同方法。

你将了解：

*   **适配器微调 (Adapter Tuning):** 了解如何在预训练模型层中插入小型、可训练的适配器模块。我们将介绍它们的架构和实现细节。
*   **前缀微调 (Prefix Tuning):** 考察如何通过向输入或隐藏状态添加可调的连续向量前缀，从而调整模型行为，同时不修改核心权重。
*   **提示微调 (Prompt Tuning):** 研究侧重于学习软提示嵌入的技术，这提供了一种高效指导模型输出的方式。

在本次概述中，我们将分析每种方法的架构差异、引入的可训练参数数量以及典型的性能特点。我们还将比较它们在训练和推理期间的内存使用和计算需求，为在特定限制和目标下选择合适的技术提供参考。本章最后将以一个实现适配器微调的实践练习结束。
