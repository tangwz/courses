---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm"
sourceId: 611
chapter: "long-short-term-memory-lstm"
title: "长短期记忆 (LSTM) 网络"
order: 5
description: "学习长短期记忆（LSTM）网络的架构和机制，该网络旨在克服RNN训练中的挑战。"
hasQuiz: true
---

我们之前讨论了训练简单循环神经网络（RNN）的困难之处，特别是梯度消失和梯度爆炸问题。这些问题使得基本RNN难以捕捉序列中相距较远元素之间的依赖关系。

本章介绍长短期记忆（LSTM）网络，这是一种专门的RNN架构，旨在克服这些局限。我们将观察让LSTM能够选择性地记忆或遗忘长序列中信息的核心组成部分。

您将学到：

*   调节网络内信息流动的*门控机制*的原理。
*   LSTM单元的详细结构，包括**遗忘门**、**输入门**和**输出门**。
*   **细胞状态**如何作为信息的传送带，使其在网络中传输时极少衰减。
*   控制LSTM单元内部更新的数学运算（$ \sigma $，$ \tanh $）。
*   为什么LSTM在需要建模长距离依赖关系的任务中，通常比简单RNN更有效。

在本章结束时，您将理解LSTM单元的内部工作原理，并认识到它们在现代序列建模中的重要性。
