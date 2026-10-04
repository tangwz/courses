---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic"
sourceId: 717
chapter: "advanced-policy-gradients-actor-critic"
title: "高级策略梯度与 Actor-Critic 方法"
order: 3
description: "学习高级策略梯度和 Actor-Critic 方法，包括 A2C、A3C、DDPG、TRPO 和 PPO，适用于连续和离散动作空间。"
hasQuiz: false
---

基础策略梯度方法，例如 REINFORCE，其梯度估计常存在高方差问题，导致学习缓慢或不稳定。本章介绍 Actor-Critic 方法，这是一类旨在应对此局限性的算法。其核心思想是维持两个组成部分：一个 *actor*（行动者），它学习策略 $\pi(a|s)$；以及一个 *critic*（评估者），它学习价值函数（例如 $V(s)$ 或 $Q(s,a)$）来评估 actor 的行动并提供低方差的梯度信号。

你将学习在此框架基础上的一些重要进展：

*   **方差降低**：例如使用基线以及 Advantage Actor-Critic (A2C/A3C) 来稳定策略更新的方法。
*   **优势估计**：理解广义优势估计 (GAE) 以平衡偏差与方差。
*   **连续控制**：例如深度确定性策略梯度 (DDPG) 等适用于连续动作空间的算法。
*   **策略优化稳定性**：确保更可靠策略改进的方法，具体而言是信任区域策略优化 (TRPO) 和近端策略优化 (PPO)。
*   **最大熵强化学习**：Soft Actor-Critic (SAC)，一种采用熵最大化以获得更好试探能力和稳定性的离策略方法。

在本章结束时，你将理解这些高级算法的原理，并准备好实现它们以解决更复杂的强化学习问题。
