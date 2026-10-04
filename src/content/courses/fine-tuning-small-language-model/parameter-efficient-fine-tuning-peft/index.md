---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-4-parameter-efficient-fine-tuning-peft"
sourceId: 1504
chapter: "parameter-efficient-fine-tuning-peft"
title: "参数高效微调 (PEFT)"
order: 4
description: "应用 LoRA 和 QLoRA 等参数高效微调技术，在有限的显存下训练模型。"
hasQuiz: false
---

在训练过程中更新神经网络的所有权重需要消耗大量计算资源。以运行标准监督式微调的典型硬件配置为例。优化器状态、梯度和模型参数会迅速耗尽可用的 GPU 显存。即使是小型语言模型，这一过程也经常超过单张消费级显卡的显存限制。

参数高效微调 (PEFT) 解决了这些硬件限制。PEFT 不再修改每个参数，而是冻结基础模型，并在特定层中注入少量可训练权重。这减少了显存占用，同时保留了让模型学习新指令的能力。

实现这一目标的一种常用方法是低秩自适应 (LoRA)。通过使用矩阵分解，LoRA 使用两个较小的矩阵来近似权重更新。如果原始权重矩阵是 $W$，则更新表示为矩阵 $A$ 和 $B$ 的乘积。更新后的权重使用以下等式计算：

$$W_{new} = W + AB$$

这种数学方法大幅减少了训练期间需要更新的参数总数。

在本章中，你将学习如何实施这些技术，在严格的内存限制下训练模型。你将了解 LoRA 的数学原理以及它是如何减少可训练参数的。你还将使用量化 LoRA (QLoRA) 以 4 位精度加载基础模型，在不降低性能的情况下进一步减少显存使用。

完成本模块后，你将学会如何配置目标模块、调整秩参数，并编写将完整的 PEFT 配置应用于基础语言模型所需的 Python 代码。
