---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-5-advanced-autoencoder-architectures"
sourceId: 652
chapter: "advanced-autoencoder-architectures"
title: "进阶自动编码器结构"
order: 5
description: "考察复杂的自动编码器变体，包括卷积、循环、对抗、VQ-VAEs和基于Transformer的模型。"
hasQuiz: false
---

之前的章节已经介绍了自动编码器的核心思想，包括经典结构、正则化方法以及变分自动编码器（VAEs）的生成能力。虽然这些奠定了坚实的基本知识，但要高效处理图像或序列等特定数据类型，或实现特定的潜在空间属性，通常需要针对具体任务调整其结构。

本章将介绍几种此类进阶自动编码器设计。我们将涉及：
*   **卷积自动编码器 (CAEs)** 专为处理图像等数据中的空间层级结构而优化。
*   **循环自动编码器 (RAEs)** 适用于时间序列或文本等序列输入。
*   **对抗自动编码器 (AAEs)** 采用对抗训练，以在潜在空间上施加期望的先验分布。
*   **向量量化变分自动编码器 (VQ-VAEs)** 借助码本，使用离散而非连续的潜在空间。
*   概述与现代序列处理相关的**基于 Transformer** 的自动编码方法。

对于每种结构，我们将讨论其构建方式，与更简单的自动编码器相比设计修改的缘由，以及其常见的应用范围。还将提及部分选定结构的实际实现细节。
