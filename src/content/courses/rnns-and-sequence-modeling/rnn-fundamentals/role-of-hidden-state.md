---
course: "rnns-and-sequence-modeling"
chapter: "rnn-fundamentals"
lesson: "role-of-hidden-state"
sourceId: 2530
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/role-of-hidden-state"
title: "隐藏状态的作用"
description: "理解隐藏状态如何充当记忆，从之前的步骤传递信息。"
order: 3
plots: []
sourceHash: "806e2434ea9c337e05a03389001cb0073106bbc8f58b72566149ce2d57406abb"
sourceCorrections: []
---

前馈网络，常见于机器学习 (machine learning)中，对每个输入样本进行独立处理。如果你向这类网络输入单词“hot”，然后是单词“dog”，网络没有固有的方式来得知“dog”是跟在“hot”后面的。它将它们视为独立的事件。这在处理序列时是一个主要限制，因为序列中顺序和上下文 (context)是基础的。句子、股票价格或音符的意义来自于它们相对于其他部分的相对位置。

隐藏状态（$h_t$）在这里变得重要。它是允许循环神经网络 (neural network) (RNN)克服前馈网络记忆限制的主要思想。你可以将隐藏状态视为网络的**记忆**。在每个时间步 $t$，RNN 不仅处理当前输入 $x_t$；它还会纳入来自*前一个*隐藏状态 $h_{t-1}$ 的信息。

回忆隐藏状态的核心计算：

$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$

请注意这个重要项 $W_{hh}h_{t-1}$。它将前一个隐藏状态 $h_{t-1}$ 中总结的信息（通过权重 (weight)矩阵 $W_{hh}$ 转换后）直接引入当前隐藏状态 $h_t$ 的计算中。因为 $h_{t-1}$ 本身是使用 $h_{t-2}$ 计算的，$h_{t-2}$ 使用 $h_{t-3}$ 计算，依此类推，所以当前隐藏状态 $h_t$ 成为整个先前输入序列（$x_0, x_1, ..., x_t$）的函数。

本质上，隐藏状态充当了网络在当前时间步之前“看到”的一切的持续概括或压缩表示。它向前传递上下文信息，使得网络在时间 $t$ 的输出 $y_t$ 不仅受到当前输入 $x_t$ 的影响，还受到之前输入的影响。

> 隐藏状态 $h$ 充当时间步之间的连接。来自输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 的信息合并以形成当前隐藏状态 $h_t$，后者随后影响输出 $y_t$ 并传递到下一个时间步 $t+1$。权重矩阵（$W_{xh}, W_{hh}, W_{hy}$）决定了这些转换。

需要了解的是，这种记忆并非完美或无限。隐藏状态通常是一个固定大小的向量 (vector)。随着网络处理更长的序列，将所有相关的过去信息概括到这个固定大小的表示中变得具有挑战性。来自遥远过去的信息可能会被稀释或被最近的输入覆盖。这个限制是形成 LSTM 和 GRU 等更高级架构的一个重要因素，我们将在稍后介绍它们。

然而，对于许多涉及中短期依赖关系的任务，简单 RNN 的隐藏状态机制提供了前馈网络所缺乏的必要记忆。它是允许 RNN 学习在序列数据中随时间展现的模式和关联的核心组成部分。如果没有隐藏状态在步骤之间传播信息，RNN 实际上会退化为一个标准前馈网络，失去对序列建模的能力。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本教材详细解释了循环神经网络，涵盖了隐藏状态的功能、计算方法及其局限性。
- [Recurrent Neural Networks (CS231n Lecture Notes)](http://cs231n.stanford.edu/2023/handouts/lecture10-rnn.pdf) — Stanford University (CS231n course staff) (2023)
  这些讲义清晰易懂地介绍了循环神经网络，阐明了隐藏状态的运作方式及其在序列处理中的价值。
- [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley-Blackwell; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  这篇基础论文描述了简单循环神经网络（SRN）及其利用上下文单元（隐藏状态）来获取序列输入表示的方法。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson Education
  这本书中关于循环神经网络的章节对隐藏状态及其在处理序列数据中的作用提供了清晰的解释。
