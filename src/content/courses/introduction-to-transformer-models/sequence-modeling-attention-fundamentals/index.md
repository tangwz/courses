---
course: "introduction-to-transformer-models"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals"
sourceId: 660
chapter: "sequence-modeling-attention-fundamentals"
title: "序列建模与注意力机制基础"
order: 1
description: "回顾 RNN 在序列建模中的难点，并引入注意力机制作为一种解决方案。"
hasQuiz: true
---

本章侧重于理解 Transformer 架构所需的基本知识。我们首先考察涉及序列数据的常见任务，例如机器翻译或文本生成，以及在有效处理此类信息（尤其是在长序列情况下）时固有的难点。

为了建立一个基准，我们将简要回顾循环神经网络（RNNs），这是一种处理序列数据的标准方法。随后，我们将讨论 RNN 的实际局限性，包括捕捉长距离依赖和计算瓶颈等问题。

此次讨论为引入注意力机制的核心思想奠定基础。你将了解到注意力机制如何为模型提供一种方式，使其在生成输出时选择性地关注输入序列的相关部分。我们将介绍如何使用查询、键和值（$Q$、$K$、$V$）这些思想来计算注意力分数的高层工作原理，以及这些分数如何用于创建加权上下文向量，这些是本课程后续章节中讨论的更高级机制的构成要素。

在本章结束时，你将理解：
*   序列到序列任务带来的挑战。
*   RNN 的基本运作方式和局限性。
*   注意力机制的基本原理。
*   计算注意力分数和上下文向量的一般过程。
