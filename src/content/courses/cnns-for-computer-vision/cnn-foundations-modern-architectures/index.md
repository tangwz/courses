---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures"
sourceId: 604
chapter: "cnn-foundations-modern-architectures"
title: "回顾CNN核心与现代架构"
order: 1
description: "回顾CNN基本知识并学习ResNet、Inception、DenseNet和EfficientNet等现代架构。了解设计选择。"
hasQuiz: false
---

本章首先巩固卷积神经网络（CNN）的核心构成。我们假定您已熟悉基本知识，但仍将简要回顾卷积层、池化操作（例如 $max-pooling$、$average-pooling$）和激活函数（例如 $ReLU(x) = max(0, x)$）等构成部分。

在此之上，我们将审视CNN架构的发展演变。我们将追溯其从早期有影响力的模型到ResNet、Inception、DenseNet和EfficientNet等当代设计的发展历程。一些主要创新点将得到分析，包括：

*   残差连接，常表示为 $$ y = \mathcal{F}(x) + x $$
*   网络中的网络原理
*   密集连接模式
*   复合缩放策略

您将了解这些架构背后的设计选择以及所涉及的权衡，同时考虑计算成本和参数效率等因素。本章最后提供使用标准深度学习框架实现这些模型的实用指导和练习。
