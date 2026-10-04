---
course: "introduction-to-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks"
sourceId: 527
chapter: "data-preparation-neural-networks"
title: "为神经网络准备数据"
order: 2
description: "学习格式化、缩放和编码数据的方法，以实现最佳网络训练。"
hasQuiz: true
---

神经网络处理数字数据，它们的有效学习能力受输入数据的格式和特性影响很大。原始数据集很少能直接使用。本章介绍准备和组织数据以进行最佳神经网络训练的必要步骤。

您将了解如何表示不同类型的输入特征，使用归一化（$x' = \frac{x - min(x)}{max(x) - min(x)}$）和标准化（$x' = \frac{x - \mu}{\sigma}$）等方法对数值数据进行缩放的重要性，以及将分类特征转换为适合网络使用的数值格式（例如独热编码）的方法。此外，我们还将介绍如何将数据分成批次以在训练期间高效处理，以及将数据集划分为训练集、验证集和测试集以进行模型开发和可靠评估的常用做法。本章结束时，您将明白如何将原始数据转换为结构化格式，以促进神经网络有效学习。
