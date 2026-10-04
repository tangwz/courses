---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning"
sourceId: 726
chapter: "multi-agent-reinforcement-learning"
title: "多智能体强化学习"
order: 6
description: "学习多智能体环境下的强化学习。涵盖非平稳性等挑战以及合作/竞争算法。"
hasQuiz: false
---

许多顺序决策问题都包含多个彼此影响的智能体，从机器人团队和自动驾驶车辆，到经济建模和博弈游戏。单智能体强化学习方法在这些情况下往往力有不逮，因为每个智能体的最佳策略都依赖于其他智能体的行为。本章将介绍多智能体强化学习（MARL），把强化学习的原则推广到多个智能体的环境中。

我们将首先对多智能体问题进行形式化，通常使用随机博弈（或马尔可夫博弈）的框架。多智能体强化学习中的一个主要难题是*非平稳性*：从任何单个智能体的角度来看，当其他智能体调整其策略时，环境动态会发生变化。我们将研究解决这些及其他挑战的不同方法，对比集中式训练方法与分布式执行策略（CTDE）。

你将学习到：

*   多智能体强化学习独有的基本挑战，主要是非平稳性。
*   不同的多智能体强化学习模式，如独立学习、参数共享和 CTDE。
*   针对合作任务设计的具体算法，例如价值分解网络（VDN）和 QMIX。
*   适用于混合合作-竞争场景的算法，例如多智能体深度确定性策略梯度（MADDPG）。
*   与智能体通信相关的内容。

到本章结束时，你将理解多智能体强化学习的核心问题，并熟悉几种用于训练多个彼此影响的智能体的常用算法。
