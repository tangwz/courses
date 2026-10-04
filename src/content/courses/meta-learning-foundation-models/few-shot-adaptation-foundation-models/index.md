---
course: "meta-learning-foundation-models"
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models"
sourceId: 853
chapter: "few-shot-adaptation-foundation-models"
title: "基础模型的少量样本适应策略"
order: 5
description: "研究在标准元学习方法之外，以适应数据量有限的大型模型的方法。"
hasQuiz: false
---

前面的章节详细介绍了多种元学习算法，但将其直接应用于大型基础模型通常会遇到显著的计算障碍。现代大型语言模型和视觉转换器的规模使得适应技术变得必要，这些技术需在少量样本情境下具备数据高效性，并且在计算上可行。本章介绍实用的策略，专门用于适应这些大型预训练模型，包括补充或替代前面讨论的元学习方法。

你将了解参数高效微调（PEFT）方法，这些方法侧重于在适应过程中仅修改模型参数的一小部分。涵盖的主要技术包括：

*   **适配器模块：** 将小型可训练模块注入固定基础模型的层中。
*   **低秩适应（LoRA）：** 通过学习低秩更新来适应权重矩阵。
*   **提示调整与前缀调整：** 学习添加到输入序列或层激活之前的连续嵌入。

本章分析这些方法的运作原理，比较它们与元学习算法的性能特点和计算需求，并考虑混合方法。你还将获得动手经验，通过实现LoRA来在一个少量样本任务上适应基础模型。
