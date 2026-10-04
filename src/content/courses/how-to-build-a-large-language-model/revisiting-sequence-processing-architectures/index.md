---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures"
sourceId: 1081
chapter: "revisiting-sequence-processing-architectures"
title: "回顾序列处理架构"
order: 3
description: "简要回顾循环架构（RNN、LSTM、GRU），以便与 Transformer 模型进行对比。"
hasQuiz: false
---

为了理解 Transformer 架构的创新之处，首先了解其之前的序列处理模型结构是有帮助的。本章简要回顾循环神经网络 (RNN) 及其更为复杂的变体：长短期记忆 (LSTM) 网络和门控循环单元 (GRU)。

我们将考察：
*   RNN 中使用隐藏状态进行序列处理的核心思想。
*   简单 RNN 面临的难题，例如由于梯度消失而难以捕捉长期依赖关系。
*   LSTM 和 GRU 中的门控机制如何被设计来缓解这些问题。
*   这些循环架构在序列到序列 (seq2seq) 任务中的应用。

此次回顾为理解为何注意力机制和 Transformer 架构在序列数据建模中代表了重大转变提供了背景，我们将在后续章节中进行讲解。
