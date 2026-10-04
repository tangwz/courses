---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-4-transformer-architecture"
sourceId: 1082
chapter: "transformer-architecture"
title: "Transformer 架构"
order: 4
description: "详细介绍 Transformer 模型的组成部分，包括自注意力、多头注意力以及位置编码。"
hasQuiz: false
---

在上一章中回顾了循环架构后，我们现在来看 Transformer 模型。循环模型逐个处理序列中的标记，这对并行计算以及建立长距离关联构成了难题。Transformer 架构通过完全用注意力机制替代循环结构，解决了这些局限。

本章将讲解 Transformer 的结构。你会了解到：

*   **缩放点积注意力**机制，它是使用查询、键和值（$Q$、$K$、$V$）的核心组成部分。
*   **多头注意力**如何让模型能够同时关注来自不同表示子空间的信息。
*   引入序列顺序的方法，即使用**位置编码**，因为自注意力本身具有排列不变性。
*   **编码器和解码器堆叠**的结构，它由注意力层和前馈层构成。
*   **层归一化和残差连接**在稳定深度网络训练中的作用。

本章结束后，你将了解使 Transformer 在序列建模任务中有效运作的基本组成部分。
