---
course: "lora-peft-efficient-llm-training"
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-1-fine-tuning-efficiency-foundations"
sourceId: 815
chapter: "fine-tuning-efficiency-foundations"
title: "回顾微调与效率的必要性"
order: 1
description: "了解全量 LLM 微调的瓶颈以及参数高效微调（PEFT）的动因。"
hasQuiz: false
---

大型语言模型（LLMs）微调是一种常用方法，用于使其适应特定任务。然而，修改包含数十亿参数的模型的所有参数（这些参数通常表示为权重矩阵 $W$），会产生大量的计算和内存开销。本章考察了伴随全量微调而来的这些难题。

您将了解到：
*   全量微调具体的内存、计算和存储需求。
*   促使参数高效替代方法不可或缺的因素。
*   奇异值分解（SVD）的数学背景，这与理解 LoRA 等方法有关，LoRA 常用于低秩近似，例如 $$M \approx U \Sigma V^T$$。
*   各种参数高效微调（PEFT）方法的分类。

这些基础内容将说明为何 PEFT 方法变得不可或缺，并介绍本课程中涵盖的各类技术。
