---
course: "intro-to-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods"
sourceId: 490
chapter: "introduction-policy-gradient-methods"
title: "策略梯度方法简介"
order: 8
description: "了解策略梯度方法，这是一种直接优化策略的强化学习方法。"
hasQuiz: true
---

到目前为止，我们一直专注于基于价值的强化学习方法。我们学习了如何估计状态 ($V(s)$) 或状态-动作对 ($Q(s,a)$) 的价值，并基于这些价值来得出策略。本章将介绍一种不同的方法：策略梯度方法。

使用策略梯度方法，我们直接学习一个参数化的策略，记作 $\pi_\theta(a|s)$。我们不先估计价值函数，而是旨在优化策略参数 $\theta$ 以最大化预期回报。这种方法在具有连续动作空间的环境中或当我们希望学习随机策略时特别适用。

在本章中，你将学习到：

*   直接参数化和优化策略的基本思想。
*   策略梯度定理背后的主要思想，它为这些方法提供了理论基础。
*   REINFORCE 算法，一种基础的蒙特卡洛策略梯度技术。
*   如何使用基线来帮助降低策略梯度估计中内在的方差。
*   Actor-Critic 方法的简要介绍，这些方法结合了基于价值和基于策略学习的特点。
*   基于策略和基于价值方法之间的权衡。

我们将从原理上讲解这些算法，并指导你实现一个基本的 REINFORCE 智能体。
