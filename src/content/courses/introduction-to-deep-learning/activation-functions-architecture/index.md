---
course: "introduction-to-deep-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture"
sourceId: 941
chapter: "activation-functions-architecture"
title: "激活函数与网络结构"
order: 2
description: "学习激活函数（Sigmoid, Tanh, ReLU）、网络层（输入, 隐藏, 输出）以及设计前馈网络。"
hasQuiz: true
---

在上一章中，我们确立了多层感知机（MLP）的思想，作为克服单层模型局限性的一种方法。本章将侧重于决定这些网络如何运作和学习的内部机制和结构设计选择。

您将了解激活函数，它们是神经元内部的非线性组成部分，使多层感知机能够模拟复杂的关系。我们将介绍Sigmoid、Tanh和ReLU（$f(x) = \max(0, x)$）等标准函数，考察它们的数学特性、优点和缺点。理解这些函数对于控制网络中信息和梯度的流动非常重要。

此外，我们将考察前馈网络的结构，阐明输入层、隐藏层和输出层的作用。我们将讨论设计网络结构时的考量因素，例如选择层数和每层单元数，为构建有效的模型打下基础。实际例子将演示如何实现和比较不同的激活函数。
