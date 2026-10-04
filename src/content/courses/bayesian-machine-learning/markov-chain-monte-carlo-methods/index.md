---
course: "bayesian-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods"
sourceId: 735
chapter: "markov-chain-monte-carlo-methods"
title: "马尔可夫链蒙特卡洛方法"
order: 2
description: "实现和分析用于贝叶斯推断的高级MCMC算法，包括HMC和NUTS。"
hasQuiz: false
---

后验分布的计算，通常表示为 $p(\theta | D)$，是贝叶斯分析的核心。然而，这些分布常常涉及高维积分，它们没有解析解，使得直接计算不可行。马尔可夫链蒙特卡洛 (MCMC) 方法提供了一个强大的计算工具集，通过从目标后验分布中生成样本来应对这一难题，使我们能够对其进行近似并计算相关统计量。

本章涵盖了MCMC技术在贝叶斯推断中的理论与实践。我们将从蒙特卡洛积分的基础思想以及保证这些方法有效的马尔可夫链必要理论开始。您将学习实现和分析几种主要的算法，包括：

*   Metropolis-Hastings算法及其变体。
*   吉布斯采样，尤其适用于具有特定条件结构的模型。
*   哈密顿蒙特卡洛 (HMC)，这是一种基于梯度的算法，能够高效处理复杂的概率空间。
*   No-U-Turn采样器 (NUTS)，是HMC的一种自适应扩展。

此外，我们将探讨评估MCMC模拟是否正确收敛并高效运行的实际需要。这包括应用标准的收敛诊断方法。一个实践环节将指导您使用现代概率编程库实现高级MCMC采样器。
