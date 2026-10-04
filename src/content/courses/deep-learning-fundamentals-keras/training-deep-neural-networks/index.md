---
course: "deep-learning-fundamentals-keras"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks"
sourceId: 928
chapter: "training-deep-neural-networks"
title: "训练深度神经网络"
order: 3
description: "理解模型训练的过程，包括编译、损失函数、优化器和评估。"
hasQuiz: true
---

在使用Keras层定义了神经网络的架构后，下一步是让它从数据中学习。本章主要讲解训练过程的机制。

你将学习如何通过*编译*模型来准备训练，这包括选择一个损失函数来衡量误差，一个优化算法来更新模型的权重，以及用于监控性能的指标。我们将介绍梯度下降及其变体（例如Adam、SGD）等重要思想，反向传播的原理（网络如何学习调整），以及使用Keras的`fit()`方法进行实际操作。

我们还将解释$epochs$和$batch$ $size$等必要的训练参数，使用验证数据来监控进度的重要性，最后，如何使用`evaluate()`方法评估模型在未见过数据上的表现。到本章结束时，你将理解如何有效地定义Keras模型并对其进行训练的完整流程。
