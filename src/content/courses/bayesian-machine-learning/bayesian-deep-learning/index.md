---
course: "bayesian-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning"
sourceId: 746
chapter: "bayesian-deep-learning"
title: "贝叶斯深度学习"
order: 6
description: "将贝叶斯方法与深度学习结合。涵盖贝叶斯神经网络（BNNs）、不确定性、VAEs以及实践训练。"
hasQuiz: false
---

深度学习模型在各种复杂任务中展现出很大的成功，但标准实现通常缺少可靠的不确定性估计。贝叶斯方法为处理不确定性提供了一个正式的框架。本章将介绍将贝叶斯原理与深度学习架构相结合的方法。

你将学习如何通过在网络参数（如权重 $w$）上设置先验分布而非使用点估计来构建贝叶斯神经网络（BNNs）。我们将解决在这些高维模型中进行推断所面临的挑战，重点在于计算或近似后验分布 $p(w | \mathcal{D})$。

主要内容包括：

*   应用高级马尔可夫链蒙特卡罗（MCMC）方法，例如随机梯度哈密顿蒙特卡罗（SGHMC），以从BNN后验分布中采样。
*   使用可扩展的变分推断（VI）技术，例如反向传播贝叶斯（Bayes by Backprop），以近似后验分布。
*   量化并区分预测中不同类型的不确定性（偶然不确定性和认知不确定性）。
*   从概率建模的角度理解变分自编码器（VAEs）。
*   分析常用的正则化技术（例如 dropout）与近似贝叶斯推断之间的联系。
*   训练和评估BNN时的实践考量。

我们将涵盖构建和应用这些模型所需的理论原理和实际实现细节。
