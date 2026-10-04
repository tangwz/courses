---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-5-model-based-reinforcement-learning"
sourceId: 723
chapter: "model-based-reinforcement-learning"
title: "基于模型的强化学习"
order: 5
description: "了解智能体如何学习环境的模型并运用它进行规划，内容涵盖Dyna-Q、MCTS集成以及MPC的理念。"
hasQuiz: false
---

迄今为止，我们主要关注了直接通过互动学习的无模型算法。本章将转向研究基于模型的强化学习。其核心思想是让智能体建立一个内部模型来描述环境的运作方式。这包括学习近似的状态转移概率（通常表示为 $P(s'|s, a)$）以及预期奖励函数 $R(s, a, s')$。

一旦模型被习得，即使是不完美的模型，智能体也可以在内部运用它进行规划或模拟经验，这可能促使真实互动的使用效率更高。我们将考察学习这些动态模型的方法，以及如何将它们与规划结合。主要内容包括Dyna-Q架构，运用已学模型进行轨迹采样，蒙特卡洛树搜索（MCTS）的原理及其集成方式，以及与模型预测控制（MPC）的联系。我们还将考虑与模型准确性以及规划计算成本相关的实际问题。在本章结束时，你将熟悉基于模型的强化学习所使用的原理、技术和常用方法。
