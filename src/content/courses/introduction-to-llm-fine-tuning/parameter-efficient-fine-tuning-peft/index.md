---
course: "introduction-to-llm-fine-tuning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft"
sourceId: 1266
chapter: "parameter-efficient-fine-tuning-peft"
title: "参数高效微调 (PEFT)"
order: 4
description: "学习现代、高效的LLM微调技术，例如LoRA和QLoRA，以调整模型并降低计算成本。"
hasQuiz: false
---

上一章讲解了全参数微调，这种办法会更新模型中的每个权重。这种办法虽然有效，但随着模型参数量达到数十亿，其计算成本变得难以承受。对GPU内存和计算能力的高要求，使全参数微调在许多实际应用中无法实现。

本章介绍参数高效微调 (PEFT)，这是一系列旨在以显著更少计算资源调整大型模型的技术。我们将重点介绍低秩适应 (LoRA)，它会冻结原始模型权重并注入更小、可训练的矩阵。LoRA并非更新庞大的原始权重矩阵 $W_0$，而是学习一个低秩更新，$$ \Delta W = BA $$，其中 $A$ 和 $B$ 中的参数数量远小于 $W_0$。你将学习如何使用Hugging Face的`PEFT`库来实现这一点。

我们还将考察量化如何进一步减少内存占用，从而引出像QLoRA这样的方法。本章最后将进行PEFT与全参数微调的比较分析，明确两者在性能和资源使用上的权衡。完成本章后，你将掌握实用技能，能够在消费级硬件上将现代、高效的微调方法应用于超大型模型。
