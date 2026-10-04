---
course: "deep-learning-fundamentals-keras"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns"
sourceId: 934
chapter: "recurrent-neural-networks-rnns"
title: "循环神经网络 (RNN)"
order: 5
description: "了解 RNN 和 LSTM，使用 Keras 进行序列资料处理。"
hasQuiz: true
---

前几章主要处理的是顺序不是主要考量点的模型输入数据，例如分类静态图片。本章将把重点转向序列很重要的资料，例如文本中的句子或时间序列中的数值。我们会介绍循环神经网络 (RNN)，这是一种特别适合处理序列资料的神经网络结构。

你会学习 RNN 如何运作，通过维护一个内部状态（通常称为隐藏状态），这使它们在处理当前元素时，能“记住”序列中先前元素的信息。我们会从 Keras 中基础的 `SimpleRNN` 层开始，并考察其能力与不足之处，包括梯度消失问题。接着，我们会学习更复杂的变体，例如长短期记忆 (LSTM) 和门控循环单元 (GRU) 网络，这些网络旨在更有效地捕获更长距离的关联。你将获得实际操作经验，如何为这些模型准备序列资料，以及如何使用 Keras 实现 RNN 和 LSTM，用于文本分类等常见任务。
