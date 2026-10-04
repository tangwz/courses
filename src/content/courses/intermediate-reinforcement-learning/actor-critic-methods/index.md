---
course: "intermediate-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods"
sourceId: 960
chapter: "actor-critic-methods"
title: "Actor-Critic 方法"
order: 5
description: "通过Actor-Critic算法结合基于策略和基于价值方法的优点。学习A2C和A3C的架构。"
hasQuiz: true
---

在之前的章节中，我们回顾了基于价值的方法，例如学习动作价值的深度Q网络（DQN），以及直接优化策略的策略梯度方法，例如REINFORCE。这两种方法各有优缺点。基于价值的方法可以提高样本效率，但在连续动作空间中表现不佳。策略梯度方法可以自然地处理连续动作，但其梯度估计往往方差较大。

本章将介绍Actor-Critic方法，这是一类结合了这两种方法特点的算法。您将了解这些方法如何使用两个组成部分：

*   **Actor（执行者）**：负责选择动作，类似于策略梯度方法。它学习一个参数化的策略$\pi_\theta(a|s)$。
*   **Critic（评论者）**：负责评估Actor所采取的动作，类似于基于价值的方法。它学习一个价值函数，通常是状态价值函数$V_\phi(s)$或动作价值函数$Q_\phi(s, a)$。

我们将分析评论者的评估如何为执行者提供方差更小的学习信号，旨在与纯策略梯度方法相比，实现更稳定、更高效的训练。我们将学习优势Actor-Critic (A2C) 及其异步变体 (A3C) 等具体实现，侧重于它们的架构、更新规则和实际考量。到本章结束时，您将理解Actor-Critic方法背后的原理，以及它们如何解决早期技术的一些局限性。
