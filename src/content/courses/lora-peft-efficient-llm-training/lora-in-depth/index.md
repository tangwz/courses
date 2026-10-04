---
course: "lora-peft-efficient-llm-training"
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth"
sourceId: 816
chapter: "lora-in-depth"
title: "低秩适配 (LoRA) 详解"
order: 2
description: "了解 LoRA 用于 LLM 的理论依据、数学表述和实际实现。"
hasQuiz: false
---

第1章概述了微调大型语言模型所涉及的计算和内存挑战。本章将通过着重介绍低秩适配 (LoRA) 来解决这些问题，LoRA是一种特定且广泛使用的参数高效微调 (PEFT) 技术。

我们从 LoRA 背后的核心假设开始：模型权重在适配过程中产生的变化 $\Delta W$ 具有较低的“内在秩”。这意味着 $\Delta W$ 可以通过两个小得多的矩阵 $B$ 和 $A$ 的乘积来有效近似。

$$ \Delta W \approx BA $$

这里 $W \in \mathbb{R}^{d \times k}$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，并且秩 $r \ll \min(d, k)$。

在本章中，您将学到：

*   LoRA 的理论依据和数学表述。
*   LoRA 如何将权重更新分解为低秩矩阵（$A$ 和 $B$）。
*   选择秩 $r$ 和配置缩放参数 $\alpha$ 的实际考量。
*   如何在神经网络层内，特别是线性层中实现 LoRA 修改。
*   将 LoRA 集成到标准 Transformer 模块中的方法，针对注意力机制或前馈网络等特定组件。

通过本章的学习，您将理解 LoRA 的运行机制，并能够实现其基本形式，以进行高效的 LLM 微调。
