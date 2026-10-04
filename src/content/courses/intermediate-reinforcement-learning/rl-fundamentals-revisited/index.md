---
course: "intermediate-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited"
sourceId: 947
chapter: "rl-fundamentals-revisited"
title: "回顾强化学习基本原理"
order: 1
description: "对强化学习主要知识点的简洁回顾，包括马尔可夫决策过程 (MDPs)、价值迭代、策略迭代、Q-学习和SARSA，为中级主题做好准备。"
hasQuiz: true
---

在处理更高级的技术之前，有必要确保我们对强化学习（RL）的主要原理有扎实的掌握。本章旨在对强化学习学科中的主要知识进行一次集中温习。

我们将简要回顾：
*   包含智能体、环境、状态、动作和奖励的标准强化学习框架。
*   马尔可夫决策过程 (MDP)，作为序列决策问题的数学表达。
*   价值函数的主要作用，特别是状态价值函数 $V(s)$ 和动作价值函数 $Q(s, a)$。
*   贝尔曼方程，它是许多强化学习算法的构建依据：
    $$ V(s) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s] $$
*   经典的表格方法，如 Q-学习 和 SARSA，以及它们的更新机制。

最后，我们将审视这些表格方法的固有局限性，特别是在处理大型或连续状态空间时。对这些限制的理解将为后续章节中介绍的函数近似方法做好铺垫。此次复习有助于我们建立共同的认知，以便继续学习深度Q网络和策略梯度方法。
