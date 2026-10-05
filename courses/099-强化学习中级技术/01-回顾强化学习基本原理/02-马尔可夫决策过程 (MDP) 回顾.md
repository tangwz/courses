# 马尔可夫决策过程 (MDP) 回顾

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/mdp-recap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

强化学习 (reinforcement learning)常处理序贯决策问题。用于建模这些问题的标准数学框架是**马尔可夫决策过程 (MDP)**。如果您之前接触过强化学习，就会明白 MDPs 是大多数算法得以构建的重要组成部分。这里将讨论MDPs的核心组成部分。

MDP 正式描述了强化学习智能体与之交互的环境。它假设环境是完全可观察的，并满足马尔可夫性质：未来状态仅取决于当前状态和动作，而不取决于之前的状态和动作序列。

MDP 通常由包含五个元素的元组定义：$(S, A, P, R, \gamma)$。

### 1. 状态集合 ($S$)

这是智能体可能处于的所有情况的集合。状态包含了与决策过程相关的所有必要环境信息。例如，在国际象棋游戏中，状态是棋盘上所有棋子的布局。在机器人导航任务中，状态可能是机器人的坐标 $(x, y)$ 和方向。
状态空间可以是：

- **有限**: 数量有限的独立状态（如在井字棋或小型网格中）。表格方法在此类情况下通常表现良好。
- **无限或连续**: 无限或连续范围的情况（如机器人关节角度或股市价格）。在这种情况下，函数逼近变得不可或缺。

### 2. 动作集合 ($A$)

这是智能体可以采取的所有可能动作的集合。可用动作可能取决于当前状态，有时表示为 $A(s)$。与状态类似，动作空间可以是：

- **离散**: 有限的一组选择（例如，在网格中的“上”、“下”、“左”、“右”，或在交易中的“买入”、“卖出”、“持有”）。
- **连续**: 一系列动作值（例如，自动驾驶汽车的转向角或火箭发动机的推力大小）。基于值的方法，如 Q-learning，难以处理连续动作，这促使采用基于策略的方法。

### 3. 转移概率函数 ($P$)

此函数定义了环境的动态特性。它指定了智能体在状态 $s$ 中采取动作 $a$ 后，转移到新状态 $s'$ 的概率。其表达式为：
$P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)$
这种概率分布捕获了环境对智能体动作响应中固有的随机性或不确定性。马尔可夫性质体现在此处：下一个状态 $s'$ 仅取决于当前状态 $s$ 和动作 $a$。

### 4. 奖励函数 ($R$)

奖励函数定义了强化学习 (reinforcement learning)问题的目标。它指定了智能体在状态 $s$ 中采取动作 $a$ 并转移到状态 $s'$ 后获得的即时数值奖励 $r$。它可以根据情况略有不同，通常定义为：

- $R(s, a, s')$: 奖励取决于起始状态、所采取的动作和结果状态。
- $R(s, a)$: 奖励仅取决于状态和动作。

智能体的目标是最大化随时间的*累积*奖励，而不仅仅是即时奖励。奖励指导学习过程，指示哪些动作会带来期望的结果。

### 5. 折扣因子 ($\gamma$)

折扣因子 $\gamma$ 是一个介于 0 和 1 之间的值 ($0 \le \gamma \le 1$)，它决定了未来奖励的当前价值。未来 $k$ 步获得的奖励将按 $\gamma^k$ 的因子进行折扣。

- 如果 $\gamma = 0$，智能体纯粹“短视”，只关心即时奖励。
- 如果 $\gamma$ 接近 1，智能体则“远视”，几乎像重视即时奖励一样重视未来奖励。

折扣因子确保在持续任务（没有终止状态的任务）中总预期奖励保持有限，并使我们能够以数学方式处理无限奖励序列。

### MDP 交互循环

智能体与环境在 MDP 框架内的交互遵循以下循环：

1. 智能体观察当前状态 $s_t \in S$。
2. 基于 $s_t$，智能体根据其策略 $\pi(a|s)$ 选择一个动作 $a_t \in A(s_t)$。
3. 环境根据转移概率 $P(s_{t+1} | s_t, a_t)$ 转移到一个新状态 $s_{t+1}$。
4. 环境根据奖励函数 $R(s_t, a_t, s_{t+1})$ 向智能体提供奖励 $r_{t+1}$。
5. 循环从新状态 $s_{t+1}$ 重复。

> 马尔可夫决策过程中的基本交互循环。智能体观察一个状态，采取一个动作，环境以新状态和奖励作为响应。

理解这种正式结构是必要的，因为强化学习 (reinforcement learning)算法本质上是在 MDPs 中寻找最优策略的方法。当我们讨论 Q-learning、DQN、策略梯度和 Actor-Critic 方法时，它们都基于一个假设：问题可以（至少近似地）建模为 MDP。表格方法的局限性，我们接下来会简单提及，当此元组的组成部分，特别是状态空间和动作空间，变得过大或过于复杂而无法明确处理时，就会出现。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本基础教材，对强化学习进行了全面介绍，其中专门有一章介绍有限马尔可夫决策过程。
- [Markov Decision Processes: Discrete Stochastic Dynamic Programming](https://www.wiley.com/en-us/Markov+Decision+Processes:+Discrete+Stochastic+Dynamic+Programming-p-9780471727828) — Martin L. Puterman (1994)
  Publisher: John Wiley & Sons
  一本权威且内容广泛的教材，介绍了马尔可夫决策过程的理论基础和应用。
- [Markov Decision Processes (CS234 Lecture 2 Notes)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGep1vK28SAFgB9eWa7pnUgHPqRYOJdIKGrLkj7qACf5wgZEAhDuxv2llRW_7YChQT0Tp7rfMYcF4-5phrR-zPmTSoDNIqPedCgisOItIUp6SBafFYqWDkk1E6ogc244j65nMSeedikzGo-pKndkgdXooR5uzLvfnNjzqjSTnuPf3Tf6VDNTEOpiDE3) — Emma Brunskill (2023)
  Publisher: Stanford University
  来自斯坦福大学强化学习课程的讲义，提供了马尔可夫决策过程组成部分和概念的简明概述。

---

[上一节](01-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98%E8%AE%BE%E7%BD%AE.md) · [下一节](03-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0%E4%B8%8E%E8%B4%9D%E5%B0%94%E6%9B%BC%E6%96%B9%E7%A8%8B.md)
