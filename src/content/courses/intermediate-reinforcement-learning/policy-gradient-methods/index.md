---
course: "intermediate-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods"
sourceId: 956
chapter: "policy-gradient-methods"
title: "策略梯度方法"
order: 4
description: "了解策略梯度方法，这类方法直接优化策略参数，无需显式进行价值函数估计。"
hasQuiz: true
---

前几章着重介绍了像深度Q网络（DQN）这样的方法，它们学习在状态下采取行动的价值$Q(s, a)$。尽管有效，但这些基于价值的方法在某些情况下可能会遇到困难，例如在连续动作空间的环境中，或者当随机策略本身是必要时。

本章将介绍策略梯度方法，这是一种截然不同的策略。在这里，我们直接学习一个参数化的策略$\pi(a|s; \theta)$来选择行动，而不依赖中间的价值函数估计来决定行动。我们将首先讨论基于价值方法的局限性，正是这些局限性促使了这种替代方法的出现。

接着，我们将阐述策略梯度背后的主要思想：调整策略参数$\theta$以最大化预期回报。这需要了解策略梯度定理，它是这些方法的理论依据。您将学习实现REINFORCE算法，这是一种蒙特卡洛策略梯度的基本技术。我们还将解决REINFORCE算法中常遇到的高方差挑战，并提出使用基线来提高稳定性和收敛速度的方法。本章最后将提供REINFORCE算法的实践练习。
