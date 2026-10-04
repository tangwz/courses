---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-2-the-gradient-boosting-machine"
sourceId: 1311
chapter: "the-gradient-boosting-machine"
title: "梯度提升机 (GBM)"
order: 2
description: "详细剖析梯度提升机算法。学习残差、损失函数、正则化以及GBM如何逐步工作。"
hasQuiz: false
---

上一章中，我们学习了使用AdaBoost的提升（boosting）原理，其中模型是按顺序构建的，以纠正其前一个模型的错误。本章在此基础上，将介绍梯度提升机（GBM），这是一种更通用、更灵活的提升算法。

梯度提升不是调整训练样本的权重，而是将问题设定为函数梯度下降。每个新的弱学习器不是在原始目标值上训练，而是在损失函数相对于前一个模型输出的负梯度上训练。对于使用均方误差的常见回归问题，这个梯度简化为残差，即 $y - \hat{y}$。这种方法使得梯度提升能够与任何可微分的损失函数配合使用，从而适用于多种不同任务。

在本章中，我们将剖析GBM算法的工作机制。你将学到：

*   梯度提升如何使提升框架更具通用性。
*   梯度的作用以及它们与残差的关联。
*   构建GBM的逐步操作步骤。
*   回归和分类中常用的损失函数。

为了巩固这些内容，本章最后会有一个实际练习，你将使用Python从头开始实现一个GBM算法的基本版本。
