# 第 2 章：循环神经网络基本原理

来源：[原章节](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals)

[返回课程目录](../README.md)

在上一章中，我们已经了解了序列数据的特点以及标准前馈网络的局限性，现在我们将重点放在专门为处理序列而设计的模型上。本章将介绍循环神经网络（RNN）的基本知识。

您将学习RNN的核心思想：即逐个元素地处理序列，同时保持内部的“记忆”或隐藏状态。我们将分析一个简单RNN单元的结构，理解在时间步 $t$ 的输入 $x_t$ 如何与前一个隐藏状态 $h_{t-1}$ 结合，以生成当前隐藏状态 $h_t$ 和一个可选输出 $y_t$。控制此过程的数学运算，通常表示为：

$$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = g(W_{hy}h_t + b_y)$$

（其中 $f$ 和 $g$ 是激活函数，如 $\tanh$ 或 sigmoid）将进行详细说明。我们将展示信息如何随时间流动，并介绍RNN的重要训练算法：随时间反向传播（BPTT），包括网络展开的思想。

在本章结束时，您将掌握基本RNN的运行原理以及其训练过程的运作方式。

## 小节

- 1. [核心思想：迭代处理序列](01-%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3%EF%BC%9A%E8%BF%AD%E4%BB%A3%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97.md)
- 2. [简单RNN架构](02-%E7%AE%80%E5%8D%95RNN%E6%9E%B6%E6%9E%84.md)
- 3. [隐藏状态的作用](03-%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 4. [RNN 单元的数学表述](04-RNN%20%E5%8D%95%E5%85%83%E7%9A%84%E6%95%B0%E5%AD%A6%E8%A1%A8%E8%BF%B0.md)
- 5. [循环神经网络中的信息流动](05-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF%E6%B5%81%E5%8A%A8.md)
- 6. [沿时间的反向传播 (BPTT)](06-%E6%B2%BF%E6%97%B6%E9%97%B4%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%20%28BPTT%29.md)
- 7. [展开网络进行训练](07-%E5%B1%95%E5%BC%80%E7%BD%91%E7%BB%9C%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.md)

章节测验：[在线测验](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/quiz)
