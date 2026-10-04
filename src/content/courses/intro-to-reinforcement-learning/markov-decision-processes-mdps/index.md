---
course: "intro-to-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps"
sourceId: 478
chapter: "markov-decision-processes-mdps"
title: "马尔可夫决策过程 (MDPs)"
order: 2
description: "了解如何使用马尔可夫决策过程 (MDPs) 对序贯决策问题进行正式建模。"
hasQuiz: true
---

在智能体、环境及其互动这些核心思想之上，我们现在需要一个正式的框架来描述强化学习所解决的问题。本章将介绍马尔可夫决策过程 (MDPs)，这是一种对序贯决策问题进行建模的常用数学工具，这类问题中的结果部分随机，部分受决策者控制。

你将了解如何定义一个MDP的主要组成部分：
*   环境可能处于的状态集合 ($S$)。
*   智能体可以采取的动作集合 ($A$)。
*   状态转移概率 ($P$)，它定义了环境的动态特性 ($P(s'|s, a)$)。
*   奖励函数 ($R$)，它规定了即时反馈 ($R(s, a, s')$)。
*   折扣因子 ($\gamma$)，它用来管理未来奖励的重要性。

我们将研究智能体的行为如何由策略 ($\pi$) 来定义，以及如何使用价值函数 ($V^\pi$ 和 $Q^\pi$) 来评估状态和状态-动作对的“优劣”。这将帮助我们理解MDP框架下强化学习的目标：即找到一个最优策略 ($\pi^*$)，使其最大化预期累积奖励。
