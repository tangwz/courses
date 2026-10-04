---
course: "nlp-fundamentals"
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro"
sourceId: 602
chapter: "nlp-sequence-models-intro"
title: "自然语言处理中的序列模型介绍"
order: 5
description: "学习用于序列文本数据的循环神经网络（RNNs）、LSTMs和GRUs的基本原理。"
hasQuiz: true
---

前几章讨论了文本表示方法，如TF-IDF，这些方法通常不考虑词语的顺序。然而，语言本身是序列化的；词语的排列方式带有重要意义。本章着重介绍旨在处理数据顺序很重要的模型。

我们将从循环神经网络（RNNs）的基本原理讲起，解释它们如何通过维护状态来处理序列。你将了解训练RNN时一个常见难题，即梯度消失问题。之后，我们将查看为解决此局限性而发展出的更复杂的架构：长短期记忆（LSTM）网络和门控循环单元（GRUs）。我们将讲解它们的核心机制，并最后讨论这些序列感知的模型如何应用于各种自然语言任务，以及一个实际的实现练习。
