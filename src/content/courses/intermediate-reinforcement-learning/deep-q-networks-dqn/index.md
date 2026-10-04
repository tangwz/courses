---
course: "intermediate-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn"
sourceId: 950
chapter: "deep-q-networks-dqn"
title: "深度Q网络 (DQN)"
order: 2
description: "了解深度Q网络（DQN）如何在强化学习中使用神经网络进行函数逼近，从而实现在高维状态空间中的学习。"
hasQuiz: true
---

第一章提到了将基本Q学习和SARSA应用于状态空间庞大或连续的问题所面临的难题。为每个可能的状态-动作对存储Q值在计算上变得不可行，并且使得泛化到未见过状态变得困难。

为了解决这个规模化问题，我们转向函数逼近。本章将介绍深度Q网络（DQN），这是一项重要进展，它使用深度神经网络来估计动作值函数$Q(s, a)$。这使得强化学习（RL）智能体能够在高维输入环境中有效学习，例如游戏屏幕图像。

我们将考察神经网络如何取代传统的Q表，以及使这种结合稳定有效的具体技术。你将了解到：

*   使用神经网络进行Q值逼近的基本思路。
*   DQN算法的核心架构和更新机制。
*   训练稳定性的重要方法：经验回放和目标网络。
*   为训练DQN定制的损失函数。
*   实现一个基本DQN智能体的分步指南。

完成本章后，你将明白DQN的运行原理，并通过为标准强化学习环境构建一个DQN智能体来获得实践经验。
