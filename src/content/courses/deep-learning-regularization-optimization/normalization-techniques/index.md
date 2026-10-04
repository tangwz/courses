---
course: "deep-learning-regularization-optimization"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-4-normalization-techniques"
sourceId: 933
chapter: "normalization-techniques"
title: "训练稳定性归一化技术"
order: 4
description: "了解批归一化如何稳定训练、加速收敛并提供正则化作用。层归一化简介。"
hasQuiz: true
---

训练深度神经网络可能具有挑战性。在训练过程中，由于前层参数的变化，后续层输入的分布也会发生偏移。这种现象，被称为内部协变量偏移，会减慢训练速度，并使其对初始化和学习率变得敏感。

本章着重介绍旨在解决这些问题的归一化方法。我们将查看：

*   **内部协变量偏移**的理念及其对训练过程的影响。
*   **批归一化**：它如何通过使用小批量统计数据（$ \mu $ 和 $ \sigma^2 $）来归一化层输入，其前向和反向传播计算，以及它在推断时的表现。
*   批归一化的**优点**，包括更快的收敛、改进的稳定性以及轻微的正则化作用。
*   在神经网络中**实现批归一化**的实际考虑。
*   简要了解**层归一化**作为另一种方法。

在本章结束时，你将明白归一化层如何有助于深度学习模型训练更稳定和高效。
