# 第 8 章：策略梯度方法简介

来源：[原章节](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods)

[返回课程目录](../README.md)

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

## 小节

- 1. [直接学习策略](01-%E7%9B%B4%E6%8E%A5%E5%AD%A6%E4%B9%A0%E7%AD%96%E7%95%A5.md)
- 2. [策略梯度定理 (理念)](02-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E5%AE%9A%E7%90%86%20%28%E7%90%86%E5%BF%B5%29.md)
- 3. [REINFORCE 算法](03-REINFORCE%20%E7%AE%97%E6%B3%95.md)
- 4. [降低方差的基线](04-%E9%99%8D%E4%BD%8E%E6%96%B9%E5%B7%AE%E7%9A%84%E5%9F%BA%E7%BA%BF.md)
- 5. [Actor-Critic 方法概述](05-Actor-Critic%20%E6%96%B9%E6%B3%95%E6%A6%82%E8%BF%B0.md)
- 6. [对比基于价值和基于策略的方法](06-%E5%AF%B9%E6%AF%94%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E5%92%8C%E5%9F%BA%E4%BA%8E%E7%AD%96%E7%95%A5%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 7. [实践：实现 REINFORCE](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20REINFORCE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/quiz)
