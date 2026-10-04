---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization"
sourceId: 607
chapter: "advanced-training-optimization"
title: "高级训练与优化方法"
order: 2
description: "掌握用于训练深度CNN的高级方法，包括优化器、学习率、正则化和归一化。"
hasQuiz: false
---

设计复杂的CNN架构，如前所述，给训练带来了难题。

深度很大的网络优化起来可能很困难，容易出现梯度消失或梯度爆炸等问题，且对超参数设置很敏感。本章将介绍有效且高效地训练这些模型所需的方法。

我们将介绍自适应调整各参数学习率的优化算法，例如AdamW和Lookahead。你将学习如何在整个训练过程中，使用各种策略管理学习率$\eta$，包括周期性方法。我们将研究标签平滑和高级的Dropout变体等正则化方法，以提高泛化能力。此外，我们还将学习标准批量归一化的替代方案，适用于深度模型的权重初始化策略，以及梯度裁剪等方法，以保持训练的稳定性。最后，我们将介绍混合精度训练等实际考量以提高速度和节省内存，同时介绍监控和调试训练循环的策略。
