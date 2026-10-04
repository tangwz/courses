---
course: "deep-learning-regularization-optimization"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers"
sourceId: 937
chapter: "adaptive-optimizers"
title: "自适应优化算法"
order: 6
description: "学习自适应学习率优化算法包括AdaGrad、RMSprop和Adam，了解它们的机制和优点。"
hasQuiz: true
---

随机梯度下降（SGD）及其动量变体等优化器相比基本梯度下降有了显著改进，但它们通常依赖于一个学习率，这个学习率对所有参数一视同仁，或按照预设方案衰减。然而，深度网络中不同的参数可能需要不同的学习率调整，根据其梯度的历史信息。

本章介绍自适应优化算法，旨在自动为每个参数独立调整学习率。我们将考察几种常用方法：

*   **AdaGrad：** 根据每个参数的梯度平方和的历史数据，来调整学习率。
*   **RMSprop：** 解决了AdaGrad学习率下降过快的问题，通过使用梯度平方的移动平均。
*   **Adam（自适应矩估计）：** 结合了动量和RMSprop的思路，存储了梯度及其平方的移动平均值。

您将学习自适应方法背后的动机，AdaGrad、RMSprop和Adam的具体更新机制，包括它们的数学基础和偏差修正技术。我们将讨论它们的优点、缺点，以及在标准深度学习框架中的实现细节。最后，我们将提供实用建议，用于为您的模型选择合适的优化器。
