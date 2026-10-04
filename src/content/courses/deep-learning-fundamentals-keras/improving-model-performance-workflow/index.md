---
course: "deep-learning-fundamentals-keras"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow"
sourceId: 936
chapter: "improving-model-performance-workflow"
title: "改进模型表现和工作流程"
order: 6
description: "学习提高模型泛化能力、正则化和高效Keras工作流程的技术。"
hasQuiz: true
---

构建和训练神经网络通常只是一个开始。确保模型在新数据上表现良好，并改进开发工作流程，是后续的必要步骤。本章侧重于诊断并处理常见的训练问题、提高模型泛化能力，以及更有效地管理训练过程的方法。

您将学会识别过拟合和欠拟合，这是模型开发中常遇到的两个问题。我们将介绍解决这些问题的方法，包括$L_1$、$L_2$和Dropout等正则化技术，以及数据增强策略。我们还会讨论使用Keras回调函数改进工作流程，例如`ModelCheckpoint`（用于在训练期间保存最佳模型）和`EarlyStopping`（以避免不必要的计算）。此外，您将学习如何正确保存和加载训练好的模型以供后续使用，并熟悉TensorBoard，用于查看训练进度和模型结构。最后，我们将提及超参数调整背后的原理。
