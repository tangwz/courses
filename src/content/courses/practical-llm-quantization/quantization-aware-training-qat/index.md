---
course: "practical-llm-quantization"
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat"
sourceId: 892
chapter: "quantization-aware-training-qat"
title: "量化感知训练 (QAT)"
order: 4
description: "了解QAT的原理与实现，以及如何在大型语言模型微调过程中模拟量化。"
hasQuiz: true
---

尽管训练后量化（PTQ）提供了一种在模型训练完成后进行量化的直接方法，但有时会导致精度明显下降，尤其是在量化到4比特整数等极低精度时。当保持精度是优先事项，且PTQ的结果不足以满足要求时，量化感知训练（QAT）提供了一种不同的方法。

QAT将量化过程直接整合到模型训练或微调阶段。通过在训练期间模拟低精度算术的影响，模型学习调整其权重，以最大限度地减少量化引起的精度损失。这通常能使最终量化模型获得比对同一原始模型应用PTQ更高的精度，特别是对于激进的量化目标。

在本章中，您将学习QAT的基础知识：

*   如何在训练循环中通过“假量化”操作模拟量化。
*   直通估计器（STE）技术，用于计算通过不可微分的量化函数（$q(x)$）的梯度。
*   使用常用的深度学习库实现QAT流程。
*   使用QAT对预训练模型进行微调的步骤。
*   比较QAT相对于PTQ的优缺点。
*   设置和运行QAT实验的实际考量。

完成本章后，您将掌握判断何时适合使用QAT以及如何实现它以生成更准确的低精度模型的知识。
