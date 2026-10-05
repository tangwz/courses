# 第 7 章：专用架构介绍

来源：[原章节](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures)

[返回课程目录](../README.md)

前馈神经网络，即多层感知机 (MLP)，提供了扎实的根基。然而，对于特定数据结构，特别是网格状数据（如图像）和序列数据（如文本或时间序列），它们并非总是最佳选择。本章将介绍为更有效地处理这些特定数据类型而发展出的专用架构。

首先，我们将介绍卷积神经网络 (CNN)。你将学习其核心组成部分，如卷积层和池化层，并理解为何这些结构在处理空间信息时表现出色。

接下来，我们将讨论循环神经网络 (RNN)。我们将讨论 RNN 如何通过引入循环和保持隐藏状态来处理序列信息，从而对序列中时间或位置上的依赖关系进行建模。本章将对 CNN 和 RNN 的设计思路、结构及基本运作方式进行基本阐述。

## 小节

- 1. [前馈网络的局限性](01-%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [卷积神经网络 (CNNs): 动因](02-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28CNNs%29-%20%E5%8A%A8%E5%9B%A0.md)
- 3. [CNN核心操作：卷积](03-CNN%E6%A0%B8%E5%BF%83%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%8D%B7%E7%A7%AF.md)
- 4. [CNN核心操作：池化](04-CNN%E6%A0%B8%E5%BF%83%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%B1%A0%E5%8C%96.md)
- 5. [典型CNN架构](05-%E5%85%B8%E5%9E%8BCNN%E6%9E%B6%E6%9E%84.md)
- 6. [循环神经网络（RNN）：缘由](06-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%EF%BC%9A%E7%BC%98%E7%94%B1.md)
- 7. [循环与隐藏状态](07-%E5%BE%AA%E7%8E%AF%E4%B8%8E%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81.md)
- 8. [基本RNN架构](08-%E5%9F%BA%E6%9C%ACRNN%E6%9E%B6%E6%9E%84.md)
- 9. [简单循环神经网络的挑战 (梯度消失/梯度爆炸)](09-%E7%AE%80%E5%8D%95%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E6%8C%91%E6%88%98%20%28%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1-%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%29.md)
- 10. [概述：LSTM与GRU](10-%E6%A6%82%E8%BF%B0%EF%BC%9ALSTM%E4%B8%8EGRU.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/quiz)
