---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-2-data-preparation-and-formatting"
sourceId: 1502
chapter: "data-preparation-and-formatting"
title: "数据准备与格式化"
order: 2
description: "学习如何组织指令数据集、应用分词技术以及管理训练中的注意力掩码。"
hasQuiz: false
---

在上一章中，我们确定了小语言模型和有监督微调的技术机制。现在，我们将重点转向数据。机器学习模型的效果与其在训练过程中处理的信息息息相关。对于微调而言，这意味着要将原始文本转换为模型架构要求的结构化格式。

原始文本无法直接输入神经网络，必须将其分解为数值表示。以标准训练数据集为例，模型需要通过分词（tokenization）将文本转换为整数数组。在进行批处理时，长度不一的序列必须进行标准化处理。我们对较短的序列进行填充（padding），使其与批次中最长的序列对齐，从而保持矩阵维度一致。如果最大序列长度为 $L$，给定输入长度为 $N$，则在数组中添加 $L - N$ 个填充标记。

本章将讲解如何为有监督微调准备自定义数据。我们将说明指令数据集的格式规则，将文本整理成结构化的指令与回答对。你将使用分词器、实施填充策略并生成注意力掩码（attention masks），以便模型在计算损失时能够自动忽略填充标记。最后，你将构建一个完整的数据流水线，用于读取自定义数据集并输出特定模型架构所需的精确张量形状。
