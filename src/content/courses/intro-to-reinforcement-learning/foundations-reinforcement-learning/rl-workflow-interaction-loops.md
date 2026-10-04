---
course: "intro-to-reinforcement-learning"
chapter: "foundations-reinforcement-learning"
lesson: "rl-workflow-interaction-loops"
sourceId: 1561
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning/rl-workflow-interaction-loops"
title: "强化学习工作流程：交互循环"
description: "描述典型强化学习设置中智能体与环境交互的循环过程。"
order: 5
plots: []
sourceHash: "92d23fc4fd6e7c88e2a1928f7554a3f6ab77b6cf75c4bef4fb8ce7ee2e7a1c5c"
sourceCorrections: []
---

智能体与环境之间的交互构成了强化学习 (reinforcement learning)过程的核心。这种交互并非一次性发生；它是一个连续的周期，常被描绘成一个循环，智能体在此循环中感知、行动并从结果中学习。理解这个工作流程对于掌握强化学习算法的运作方式非常重要。

智能体与环境的交互在强化学习中是一个连续的循环，构成强化学习过程的核心。为了理解这个交互循环，这个过程可以逐步分析。通常，时间被建模为离散的步长：$t=0, 1, 2, 3, \dots$。在每个时间步$t$，会发生以下序列：

1. **观察：** 智能体观察环境的当前状态。我们将此状态表示为 $S_t \in \mathcal{S}$，而 $\mathcal{S}$ 是所有可能状态的集合。状态表示 $S_t$ 应包含智能体做出决策所需的所有相关信息。
2. **行动选择：** 基于观察到的状态 $S_t$，智能体选择一个行动 $A_t \in \mathcal{A}(S_t)$，而 $\mathcal{A}(S_t)$ 是状态 $S_t$ 中可用行动的集合。此选择受智能体当前策略 $\pi$ 的支配。策略 $\pi(a|s)$ 定义了在状态 $s$ 时采取行动 $a$ 的概率。对于确定性策略，它将状态直接映射到行动。
3. **环境转移与奖励：** 环境接收到智能体的行动 $A_t$。基于 $S_t$ 和 $A_t$，会发生两件事：

   - 环境转移到新状态 $S_{t+1}$。这种转移由环境的动态特性决定，通常建模为概率分布 $p(s'|s, a) = P(S_{t+1}=s' | S_t=s, A_t=a)$。在许多实际情况下，智能体不明确知道此转移函数；它通过交互进行学习。
   - 环境向智能体提供一个标量奖励信号 $R_{t+1} \in \mathbb{R}$。此奖励反映了在状态 $S_t$ 中采取行动 $A_t$ 的即时结果。记住，智能体的目标通常不仅仅是最大化这个即时奖励，而是最大化随时间累积的奖励（即“回报”）。
4. **下一步：** 智能体发现自己处于状态 $S_{t+1}$，并已收到奖励 $R_{t+1}$。循环在时间步 $t+1$ 重复：观察 $S_{t+1}$，选择 $A_{t+1}$，依此类推。

这个持续的循环会生成一系列状态、行动和奖励，常被称为“轨迹”或“经验”：

$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots$

这个序列代表了智能体的交互历史。多数强化学习算法正是运用这些经验数据来学习和提升智能体的策略 $\pi$，目标是选择能够最大化预期未来累积奖励的行动。

我们可以将这个交互循环可视化：

> 强化学习交互循环。智能体观察状态（$S_t$），根据其策略（$\pi$）选择一个行动（$A_t$），环境则以奖励（$R_{t+1}$）响应并转移到新状态（$S_{t+1}$），循环持续进行。

这个循环是“回合制任务”的根本，你之前已经学过它，在其中交互自然地分解为以终止状态（例如赢得或输掉游戏）结束的序列（回合）；它也是“持续性任务”的根本，在其中交互可能无限期地进行（例如管理电网）。在回合制任务中，循环在达到终止状态时结束，并且新回合通常从某个初始状态分布开始。在持续性任务中，循环无限期运行。对于这些任务，使用因子 $\gamma \in [0, 1)$ 折扣未来奖励的想法变得尤为重要，这确保了未来奖励的总和（即回报）通常保持有限，并优先考虑更即时的奖励。

这个交互循环生成的数据，特别是以元组形式（如 $(S_t, A_t, R_{t+1}, S_{t+1})$）经历的转移，是学习的原始材料。Q学习、SARSA 或策略梯度方法（我们将在后续章节介绍）等算法处理这些经验流，以更新智能体的策略或对价值的内部估计，逐步引导智能体做出更好的决策，并实现最大化长期累积奖励的目标。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  强化学习的标准教材，详细解释了智能体-环境接口、交互循环的组成部分和基本概念。
- [Reinforcement Learning Lecture 1: Introduction to Reinforcement Learning](https://www.davidsilver.uk/teaching/) — David Silver (2015)
  Journal: University College London (UCL) Course Materials
  来自知名课程的入门讲座，清晰地概述了强化学习问题，包括智能体-环境交互模型。
- [Markov Decision Processes (MDPs)](https://cs188.org/sp17/lectures/10-mdps.pdf) — Pieter Abbeel, Dan Klein (2017)
  Journal: UC Berkeley CS188 Course Materials; Publisher: University of California, Berkeley
  讲义，提供了马尔可夫决策过程的简明解释，该过程正式建模了强化学习交互循环所描述的序列决策过程。
