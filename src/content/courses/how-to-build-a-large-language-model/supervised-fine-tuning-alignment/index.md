---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment"
sourceId: 1103
chapter: "supervised-fine-tuning-alignment"
title: "对齐微调：监督式微调 (SFT)"
order: 25
description: "学习如何使用监督技术微调预训练大型语言模型，以实现指令遵循、对话和特定任务适配。"
hasQuiz: false
---

预训练大型语言模型从海量文本语料库中习得了广泛的知识和生成能力。然而，这种预训练目标本身并不能保证模型会遵循特定指令、进行自然对话或遵守安全准则。弥合这一差距需要进行*对齐*，即调整模型以更好地符合人类意图和偏好的过程。

本章主要讲解监督式微调 (SFT)，它通常是对齐过程中的第一步。SFT 采用精心整理的提示-响应示例数据集来教导模型实现期望行为，例如遵循指令或执行特定任务。我们将探讨驱动大型语言模型对齐的目标、SFT 过程的详细机制、创建有效指令数据集的方法、适当的数据格式、训练循环和超参数的考量，以及评估 SFT 达到其对齐目标程度的技术。
