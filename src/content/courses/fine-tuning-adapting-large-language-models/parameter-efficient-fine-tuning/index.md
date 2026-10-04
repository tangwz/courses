---
course: "fine-tuning-adapting-large-language-models"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning"
sourceId: 670
chapter: "parameter-efficient-fine-tuning"
title: "参数高效微调 (PEFT) 方法"
order: 4
description: "详细研究 PEFT 技术，包括 LoRA、QLoRA、适配器、提示微调和前缀微调，以便有效调整大型语言模型。"
hasQuiz: false
---

完全微调会更新大型语言模型中的每个参数，这需要大量的计算资源和内存。这通常使得在没有专用硬件的情况下调整最大型的模型变得不切实际。本章介绍参数高效微调 (PEFT) 方法，作为一种节省资源的替代方案。

PEFT 技术只修改模型参数的一小部分，大大减少了计算开销和存储需求，同时在特定任务上通常能达到与完全微调相近的性能。

你将学习几种主要 PEFT 方法的原理和实现：

*   **低秩适应 (LoRA)：** 将可训练的低秩矩阵注入到 Transformer 层中。
*   **量化 LoRA (QLoRA)：** 将 LoRA 与量化结合，以实现更大的内存节省。
*   **适配器模块 (Adapter Modules)：** 在现有架构内插入小的、可训练的瓶颈层。
*   **提示微调 (Prompt Tuning) 和前缀微调 (Prefix Tuning)：** 学习添加到输入或隐藏状态前的连续嵌入，同时保持基本模型不变。

我们将分析每种技术背后的机制，比较它们各自的优点和权衡，并通过使用常用库进行实践。在本章末尾，你将了解如何选择和运用合适的 PEFT 方法来高效调整大型语言模型。
