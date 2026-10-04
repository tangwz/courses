---
course: "rnns-and-sequence-modeling"
chapter: "long-short-term-memory-lstm"
lesson: "addressing-rnn-limitations-gating"
sourceId: 2564
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/addressing-rnn-limitations-gating"
title: "通过门控应对循环神经网络的局限"
description: "介绍门控机制作为一种控制信息流和缓解梯度问题的方式。"
order: 1
plots: []
sourceHash: "55c04e8ac2b2767134df078375d5aaa22c5909fa269e3cc22ca44a96457a1c58"
sourceCorrections: []
---

简单循环神经网络 (neural network)（RNN）在长序列训练中面临显著困难。问题主要在于训练时梯度如何随时间反向传播 (backpropagation)（BPTT）。与早期时间步信息相关的梯度必须经过许多循环连接。在简单RNN中，这些梯度会重复乘以相同的权重 (weight)矩阵。这种重复乘法可能导致梯度指数级地缩小至接近零（梯度消失），或者指数级地增长到无法控制（梯度爆炸）。

结果是什么？简单RNN难以学习序列中相距较远元素之间的依赖关系。梯度消失问题意味着网络会有效遗忘遥远过去的旧信息，从而无法将早期输入与后期输出关联起来。梯度爆炸则导致训练不稳定，权重更新变得过大且不规律。

为应对这些局限，长短期记忆（LSTM）网络引入了一个基本思想：**门控机制**。可以把这些门看作是嵌入 (embedding)在RNN单元中的精巧阀门或开关。与简单RNN中信息相对不受控地流动（仅通过矩阵乘法和激活函数 (activation function)）不同，LSTM使用门来动态地*调节*信息流。

这些门不是固定的；它们本身就是神经网络，通常使用S型（sigmoid）激活函数（$\sigma$）。S型函数输出介于0到1之间的值。这个输出作为控制信号：

- 接近0的值表示“不让任何东西通过”或“关闭该门”。
- 接近1的值表示“让所有东西通过”或“打开该门”。
- 介于两者之间的值允许部分信息通过。

重要的一点是，网络会根据当前输入和前一个隐藏状态来*学习*如何操作这些门。这使得LSTM单元能够在每个时间步基于上下文 (context)做出决定，关于：

1. **从其内部记忆中丢弃哪些信息**。
2. **在内部记忆中存储哪些新信息**。
3. **将哪些信息作为当前时间步的隐藏状态输出**。

> 门控机制调节信息流的示意图。一个习得的控制信号通过S型（$\sigma$）激活函数，产生一个介于0到1之间的门值，然后该门值与输入信息流相乘，从而有效控制通过的信息量。

通过引入这种选择性控制，门控机制直接应对了梯度消失和梯度爆炸问题。它们使得重要信息能够跨越许多时间步向前传播而不会显著衰减（应对梯度消失），并阻止不相关信息导致过度的更新（有助于应对梯度爆炸）。这种在长时间内选择性地记忆和遗忘信息的能力是LSTM在许多序列建模任务中远比简单RNN更有效的主要原因。

在接下来的章节中，我们将了解LSTM单元中使用的具体门：即遗忘门、输入门和输出门，以及它们所调节的核心组成部分——细胞状态。

## 参考资料

- [Long Short-Term Memory](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  首次提出长短期记忆（LSTM）网络及其门控机制，以解决RNN中梯度问题。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，详细阐述了循环神经网络、梯度消失/爆炸问题以及LSTM网络的架构和功能。
- [Recurrent Neural Networks (RNNs) and LSTMs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHgJtK-odiK2ubUxgj-l78Nj-9G141ObEd3-lX1ZaJMljukCs8Xz6rFPSlpYEN5spFyqyeKGPFsbiKP0zXqmzpIfgPe5BOvDJa46ah5RSXb2-5tw7wsXlKfH5fmiWrVTzdlGvc9IohzXR8WJKyCsHLjYhxcXnlSpxdMtvxrOHByDrwDurNDbTk0IANusWJpArp9gveWc9bIYqIIMNSThPbxXQ==) — Chris Manning (2023)
  Publisher: Stanford University
  来自一所顶尖大学课程的讲义，清晰地解释了RNN的局限性、门控概念以及LSTM的结构。
