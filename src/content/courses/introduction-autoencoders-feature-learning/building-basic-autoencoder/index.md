---
course: "introduction-autoencoders-feature-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder"
sourceId: 1151
chapter: "building-basic-autoencoder"
title: "构建一个基础自动编码器"
order: 5
description: "一份实用指南，介绍如何使用Python、TensorFlow和Keras构建和训练一个简单的自动编码器以实现数据重建。"
hasQuiz: true
---

之前的章节详细介绍了自动编码器的核心原理，涵盖了其架构和学习机制。本章将从这些原理转向实际应用。在此，我们将提供一份分步指南，指导您构建您的第一个基础自动编码器。

您将学会：
*   搭建适合深度学习任务的Python环境。
*   使用PyTorch和Keras来定义和构建神经网络模型。
*   加载、理解和预处理一个常见数据集（例如MNIST），用于训练自动编码器。
*   构建简单自动编码器的编码器和解码器层。
*   配置训练过程，包括选择优化器和损失函数，例如均方误差（MSE）。MSE衡量原始输入$x$和重建输出$\hat{x}$之间的平均平方差，通常表示为 $$L(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$
*   训练自动编码器，然后评估其重建数据的能力。
*   可视化原始输入及其重建版本以评估性能。

通过学习本章，您将获得实现自动编码器和观察其数据重建能力的直接经验。这项实际练习将巩固您对这些网络如何运作的理解。
