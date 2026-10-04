---
course: "introduction-to-deep-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization"
sourceId: 943
chapter: "backpropagation-advanced-optimization"
title: "反向传播和高级优化"
order: 4
description: "理解反向传播算法、计算图以及 Momentum、RMSprop 和 Adam 等高级优化器。"
hasQuiz: true
---

在上一章中，我们明确了训练目标：通过梯度下降调整网络权重，以最小化损失函数 $L$。然而，在多层网络中有效计算关于*所有*权重的梯度 $ \nabla L $ 需要一种特定的方法。

本章将介绍反向传播算法，它是计算这些梯度的标准方法。我们将审视反向传播在微积分链式法则中的原理，并通过计算图来描绘其过程。随后，我们将超越基本的梯度下降，学习更精密的优化算法。这些算法包括 Momentum、RMSprop 和 Adam，它们有助于加速收敛，并能更有效地应对复杂的损失曲面。完成本章后，你将理解梯度是如何计算并反向传播通过网络的，以及高级优化器如何改进训练过程。
