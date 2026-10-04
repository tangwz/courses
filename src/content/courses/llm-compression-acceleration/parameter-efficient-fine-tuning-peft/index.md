---
course: "llm-compression-acceleration"
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft"
sourceId: 857
chapter: "parameter-efficient-fine-tuning-peft"
title: "参数高效微调 (PEFT) 及适配"
order: 5
description: "掌握PEFT技术，例如LoRA、QLoRA、适配器和提示微调，用于高效的大型语言模型适配。"
hasQuiz: false
---

从头训练大型语言模型需要大量资源。将这些预训练模型适配到特定的下游任务通常涉及对所有参数进行微调，这也会带来高昂的计算成本，并且如果管理多个针对特定任务的模型，还会导致存储需求巨大。

参数高效微调 (PEFT) 技术提供了一些方法，通过仅修改大型语言模型的一小部分参数，或添加少量新参数来适配它们。这种做法显著降低了微调相关的计算成本和内存占用，使得为各种应用定制大型模型成为可能，而无需完全重新训练它们或存储大量的完整副本。

本章将介绍几种突出的PEFT策略。我们将了解适配器模块的工作原理，各种基于提示的微调方法，如前缀微调 (Prefix Tuning) 和提示微调 (Prompt Tuning)，以及广泛使用的低秩适配 (LoRA)。我们还将介绍量化LoRA (QLoRA)，它通过结合量化进一步减少内存使用。您将了解这些方法的工作方式，分析它们的性能权衡，并获得实践它们的实际经验。
