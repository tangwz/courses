# 第 1 章：回顾CNN核心与现代架构

来源：[原章节](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures)

[返回课程目录](../README.md)

本章首先巩固卷积神经网络（CNN）的核心构成。我们假定您已熟悉基本知识，但仍将简要回顾卷积层、池化操作（例如 $max-pooling$、$average-pooling$）和激活函数（例如 $ReLU(x) = max(0, x)$）等构成部分。

在此之上，我们将审视CNN架构的发展演变。我们将追溯其从早期有影响力的模型到ResNet、Inception、DenseNet和EfficientNet等当代设计的发展历程。一些主要创新点将得到分析，包括：

*   残差连接，常表示为 $$ y = \mathcal{F}(x) + x $$
*   网络中的网络原理
*   密集连接模式
*   复合缩放策略

您将了解这些架构背后的设计选择以及所涉及的权衡，同时考虑计算成本和参数效率等因素。本章最后提供使用标准深度学习框架实现这些模型的实用指导和练习。

## 小节

- 1. [卷积神经网络构建模块简要回顾](01-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9D%97%E7%AE%80%E8%A6%81%E5%9B%9E%E9%A1%BE.md)
- 2. [CNN 架构的演变：从 AlexNet 到 ResNet](02-CNN%20%E6%9E%B6%E6%9E%84%E7%9A%84%E6%BC%94%E5%8F%98%EF%BC%9A%E4%BB%8E%20AlexNet%20%E5%88%B0%20ResNet.md)
- 3. [理解残差连接与跳跃架构](03-%E7%90%86%E8%A7%A3%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5%E4%B8%8E%E8%B7%B3%E8%B7%83%E6%9E%B6%E6%9E%84.md)
- 4. [Inception 模块和网络中的网络思想](04-Inception%20%E6%A8%A1%E5%9D%97%E5%92%8C%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E7%BD%91%E7%BB%9C%E6%80%9D%E6%83%B3.md)
- 5. [DenseNet：架构与连接模式](05-DenseNet%EF%BC%9A%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%BF%9E%E6%8E%A5%E6%A8%A1%E5%BC%8F.md)
- 6. [EfficientNet：模型复合缩放](06-EfficientNet%EF%BC%9A%E6%A8%A1%E5%9E%8B%E5%A4%8D%E5%90%88%E7%BC%A9%E6%94%BE.md)
- 7. [架构设计选择与权衡](07-%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1%E9%80%89%E6%8B%A9%E4%B8%8E%E6%9D%83%E8%A1%A1.md)
- 8. [现代架构构建实践](08-%E7%8E%B0%E4%BB%A3%E6%9E%B6%E6%9E%84%E6%9E%84%E5%BB%BA%E5%AE%9E%E8%B7%B5.md)
