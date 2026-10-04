---
course: "intermediate-reinforcement-learning"
chapter: "rl-fundamentals-revisited"
lesson: "value-functions-bellman-review"
sourceId: 5087
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/value-functions-bellman-review"
title: "价值函数与贝尔曼方程"
description: "重新审视状态价值函数 (V) 和动作价值函数 (Q)，以及贝尔曼期望方程。"
order: 3
plots: []
sourceHash: "f46cc7fe52428b1755375335c907a2a4525ef8774d1533b2a38746c43ef35ea7"
sourceCorrections: []
---

智能体在强化学习 (reinforcement learning)中需要一种方法来评估特定情境的优劣。价值函数在此发挥作用。它们量化 (quantization)了智能体可以获得的预期长期回报（累积折扣奖励）。价值函数主要有两种类型：状态价值函数 $V(s)$ 和动作价值函数 $Q(s, a)$。

### 状态价值函数：$V^\pi(s)$

状态价值函数，表示为 $V^\pi(s)$，代表智能体从状态 $s$ 开始并此后遵循特定策略 $\pi$ 所获得的预期回报。可以理解为回答了这个问题：“如果我遵循策略 $\pi$，处于状态 $s$ 有多好？”

其数学定义如下：
$V^\pi(s) = \mathbb{E}_\pi [ G_t | S_t = s ] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s \right]$
此处：

- $G_t$ 是从时间步 $t$ 开始的总折扣回报。
- $\gamma$ 是折扣因子 ($0 \le \gamma \le 1$)，它决定了未来奖励的现值。接近0的 $\gamma$ 使智能体“短视”（专注于即时奖励），而接近1的 $\gamma$ 使其“远见”（高度重视未来奖励）。
- $\mathbb{E}_\pi[\cdot]$ 表示期望值，假设智能体遵循策略 $\pi$。此期望值考虑了策略选择动作（如果它是随机的）的随机性以及环境转移和奖励的随机性。

### 动作价值函数：$Q^\pi(s, a)$

动作价值函数 $Q^\pi(s, a)$ 更进一步。它代表智能体从状态 $s$ 开始，采取特定动作 $a$，然后*再*在所有后续步骤中遵循策略 $\pi$ 所获得的预期回报。它回答了这个问题：“从状态 $s$ 采取动作 $a$，然后遵循策略 $\pi$ 有多好？”

其定义为：
$Q^\pi(s, a) = \mathbb{E}_\pi [ G_t | S_t = s, A_t = a ] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s, A_t = a \right]$
$Q$ 函数对决策尤其有用。如果智能体知道给定状态 $s$ 中所有可能动作的 $Q$ 值，它只需选择具有最高 $Q$ 值的动作来最优行动（或至少是根据其当前估计值采取贪婪行动）。

### 贝尔曼期望方程

价值函数遵循称为贝尔曼方程的递归关系。这些方程将状态或状态-动作对的价值分解为获得的即时奖励加上后继状态的折扣价值。

**贝尔曼期望方程（针对 $V^\pi(s)$）** 将状态 $s$ 的价值与下一状态的期望值相关联，给定策略 $\pi$：
$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$
我们来分解一下：

- $\pi(a|s)$：在策略 $\pi$ 下，在状态 $s$ 中采取动作 $a$ 的概率。
- $p(s', r | s, a)$：在给定状态 $s$ 和动作 $a$ 的情况下，转移到状态 $s'$ 并获得奖励 $r$ 的概率。这定义了环境的动态特性。
- $r + \gamma V^\pi(s')$：即时奖励 $r$ 加上下一状态 $s'$ 的折扣价值。

该方程主要表明，在策略 $\pi$ 下状态 $s$ 的价值是即时奖励加上随后出现的任何状态的折扣价值的平均值（根据 $\pi$ 采取的动作和根据环境进行的转移的平均）。

类似地，**贝尔曼期望方程（针对 $Q^\pi(s, a)$）** 将在状态 $s$ 中采取动作 $a$ 的价值与下一状态-动作对的期望值关联起来：
$Q^\pi(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$
这可以通过代入 $V^\pi(s')$ 的定义来简化：
$Q^\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$
该方程表明，在状态 $s$ 中采取动作 $a$ 的价值是预期即时奖励加上*下一*状态的预期折扣价值，假设智能体从该下一状态起继续遵循策略 $\pi$。

以下图表说明了贝尔曼期望方程所描述的这种递归关系：

> 此图表显示了当前状态 $V^\pi(s)$ 的价值如何根据策略 $\pi$ 依赖于动作价值 $Q^\pi(s, a)$。每个动作价值反过来又依赖于采取动作并获得奖励后达到的下一状态 $V^\pi(s')$ 的期望值，并根据环境的转移概率进行加权。

这些期望方程是策略评估的基础，策略评估是针对给定策略求取价值函数的过程。

### 贝尔曼最优方程

期望方程有助于评估*给定*策略，而我们在强化学习 (reinforcement learning)中的最终目标通常是找到*最佳*策略，即能够使从任何起始状态获得的预期回报最大化的策略。这是最优策略，表示为 $\pi^*$。

对应于此最优策略的价值函数是最优状态价值函数 $V^*(s)$ 和最优动作价值函数 $Q^*(s, a)$。

- $V^*(s) = \max_{\pi} V^\pi(s)$
- $Q^*(s, a) = \max_{\pi} Q^\pi(s, a)$

这些最优价值函数满足**贝尔曼最优方程**。与涉及对策略动作进行平均的期望方程不同，最优方程涉及对动作进行最大化。

**贝尔曼最优方程（针对 $V^*(s)$）** 为：
$V^*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]$
该方程表明，在最优策略下，状态的价值必须等于可能从该状态采取的*最佳*动作的预期回报。

**贝尔曼最优方程（针对 $Q^*(s, a)$）** 为：
$Q^*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')]$
该方程表明，在状态 $s$ 中采取动作 $a$ 的最优价值是预期即时奖励加上从*下一*状态 $s'$ 可获得的折扣最大 Q 值。$\max_{a'}$ 反映了从状态 $s'$ 开始，智能体将再次根据最优策略选择最佳可能动作这一事实。

如果我们知道最优动作价值函数 $Q^*(s, a)$，我们就可以轻松确定最优策略 $\pi^*$。在任何状态 $s$ 中，最优策略只需选择使 $Q^*(s, a)$ 最大化的动作 $a$：
$\pi^*(s) = \arg\max_a Q^*(s, a)$
这是相对于最优 Q 函数的贪婪策略。

直接求解贝尔曼最优方程通常是强化学习算法的目标。像价值迭代这样的方法通过迭代应用 $V^*$ 的贝尔曼最优更新来工作。对于我们稍后将考察的方法来说非常重要，Q-学习使用样本 $(s, a, r, s')$ 迭代地逼近 $Q^*(s, a)$，基于 $Q^*$ 的贝尔曼最优方程。

理解这些价值函数以及支配它们的贝尔曼方程非常重要。它们为许多强化学习算法提供了理论基础，包括我们接下来将简要回顾的表格方法以及构成本课程核心的更高级函数逼近技术。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本全面介绍强化学习理论与方法的教科书，包含对价值函数和贝尔曼方程的解释。
- [UCL Course on Reinforcement Learning: Lecture 2, Markov Decision Processes](https://www.davidsilver.uk/teaching/lec2_mdps.pdf) — David Silver (2015)
  通过讲座材料清晰地解释了马尔可夫决策过程、价值函数和贝尔曼方程。
