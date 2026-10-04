---
course: "deep-learning-regularization-optimization"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization"
sourceId: 930
chapter: "dropout-regularization"
title: "丢弃法正则化"
order: 3
description: "了解丢弃法的工作原理，以防止神经元相互依赖并减少过拟合。在神经网络中实现丢弃法。"
hasQuiz: true
---

在我们了解了 L1 和 L2 等惩罚大权重值的权重正则化方法之后，接下来我们介绍一种截然不同的防止过拟合的方法：丢弃法。这种方法在每次训练更新中，通过随机将一部分神经元输出设为零来发挥作用。这有助于避免神经元之间出现复杂的相互依赖，即神经元过度依赖特定其他神经元的情况。

本章将讲解丢弃法的工作原理，包括它在训练时如何运作，以及在推理时（测试阶段）激活值如何调整。我们将介绍常见的“倒置丢弃法”实现，讨论作为可调超参数的丢弃率 $p$，并提及在卷积网络和循环网络中应用丢弃法的考量。最后，你将通过实际例子了解如何使用标准深度学习框架将丢弃层整合到模型中。
