---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals"
sourceId: 606
chapter: "rnn-fundamentals"
title: "循环神经网络基本原理"
order: 2
description: "学习简单循环神经网络（RNN）的基本结构、运行方式和数学表示。"
hasQuiz: true
---

在上一章中，我们已经了解了序列数据的特点以及标准前馈网络的局限性，现在我们将重点放在专门为处理序列而设计的模型上。本章将介绍循环神经网络（RNN）的基本知识。

您将学习RNN的核心思想：即逐个元素地处理序列，同时保持内部的“记忆”或隐藏状态。我们将分析一个简单RNN单元的结构，理解在时间步 $t$ 的输入 $x_t$ 如何与前一个隐藏状态 $h_{t-1}$ 结合，以生成当前隐藏状态 $h_t$ 和一个可选输出 $y_t$。控制此过程的数学运算，通常表示为：

$$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = g(W_{hy}h_t + b_y)$$

（其中 $f$ 和 $g$ 是激活函数，如 $\tanh$ 或 sigmoid）将进行详细说明。我们将展示信息如何随时间流动，并介绍RNN的重要训练算法：随时间反向传播（BPTT），包括网络展开的思想。

在本章结束时，您将掌握基本RNN的运行原理以及其训练过程的运作方式。
