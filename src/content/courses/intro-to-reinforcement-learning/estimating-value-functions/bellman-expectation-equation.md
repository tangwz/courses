---
course: "intro-to-reinforcement-learning"
chapter: "estimating-value-functions"
lesson: "bellman-expectation-equation"
sourceId: 1604
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-3-estimating-value-functions/bellman-expectation-equation"
title: "贝尔曼期望方程"
description: "推导并理解 Vπ 和 Qπ 的贝尔曼期望方程。"
order: 1
plots: []
sourceHash: "abb6d49d6c1a1c6ad648d2e788a7b2e9e4b89027492c9acd9ae47d2394ad7775"
sourceCorrections: []
---

状态值函数 $V^{\pi}(s)$ 和动作值函数 $Q^{\pi}(s,a)$ 分别表示从状态 $s$ 开始，或从状态 $s$ 并采取动作 $a$ 开始，并随后遵循策略 $\pi$ 所获得的期望回报。需要一种方法来计算这些值。贝尔曼方程通过将状态或状态-动作对的值与其可能的后续状态的值联系起来，提供了一种计算机制。它们建立了一种递归关系，这对理解和解决强化学习 (reinforcement learning)问题非常重要。

**贝尔曼期望方程**具体描述了*给定*策略 $\pi$ 的这种关系。它将值函数表示为即时期望奖励加上下一个状态的折扣期望值。我们来考察一下 $V^{\pi}$ 和 $Q^{\pi}$ 这两种情况。

### V\pi(s) 的贝尔曼期望方程

回顾状态值函数的定义：
$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$
其中 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$ 是从时间 $t$ 开始的总折扣回报。我们可以将回报递归地改写为：
$G_t = R_{t+1} + \gamma G_{t+1}$
将此代入 $V^{\pi}(s)$ 的定义中：
$V^{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s]$
利用期望的线性性质，我们将其分为两部分：
$V^{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} | S_t = s] + \gamma \mathbb{E}_{\pi}[G_{t+1} | S_t = s]$
为了计算这些期望，我们需要考虑智能体可能根据其策略 $\pi(a|s)$ 采取的动作，以及由环境动力学 $p(s', r | s, a)$ 决定的可能产生的后续状态 $s'$ 和奖励 $r$。

第一项，$\mathbb{E}_{\pi}[R_{t+1} | S_t = s]$，是期望即时奖励。智能体首先根据 $\pi(a|s)$ 选择一个动作 $a$，然后环境则以根据 $p(s', r | s, a)$ 的后续状态 $s'$ 和奖励 $r$ 进行响应。对所有可能性求和：
$\mathbb{E}_{\pi}[R_{t+1} | S_t = s] = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) r$
第二项，$\gamma \mathbb{E}_{\pi}[G_{t+1} | S_t = s]$，涉及从*下一个*状态 $S_{t+1}$ 开始的期望折扣回报。期望 $\mathbb{E}_{\pi}[G_{t+1} | S_t = s]$ 对从状态 $s$ 采取的动作 $a$ 以及产生的后续状态 $s'$ 进行平均。从后续状态 $s'$ 开始的期望回报就是 $V^{\pi}(s')$。因此：
$\mathbb{E}_{\pi}[G_{t+1} | S_t = s] = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) V^{\pi}(s')$
请注意，在 $p(s', r | s, a)$ 中的奖励 $r$ 不影响 $V^{\pi}(s')$ 的值，因此我们也可以对 $p(s' | s, a) = \sum_r p(s', r | s, a)$ 求和。

将两项合并，得到 $V^{\pi}$ 的贝尔曼期望方程：
$V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V^{\pi}(s')]$

这个方程表明，在策略 $\pi$ 下处于状态 $s$ 的值，是所有动作 $a$ 的平均值（按采取这些动作的概率 $\pi(a|s)$ 加权），即期望即时奖励加上后续状态 $s'$ 的折扣期望值（对所有可能的 $s'$ 和 $r$ 进行平均，并按 $p(s', r | s, a)$ 加权）。

通常可以把这种关系看作是动作值函数 $Q^{\pi}(s,a)$。值 $V^{\pi}(s)$ 仅仅是 $Q^{\pi}(s,a)$ 的期望值，对策略 $\pi$ 可能在状态 $s$ 中选择的动作 $a$ 上的平均：
$V^{\pi}(s) = \sum_{a} \pi(a|s) Q^{\pi}(s,a)$

> 状态值 $V^{\pi}(s)$ 是动作值 $Q^{\pi}(s,a)$ 的期望值，对策略 $\pi$ 所选择的动作 $a$ 进行平均。

### Q\pi(s,a) 的贝尔曼期望方程

我们可以为动作值函数 $Q^{\pi}(s,a)$ 推导一个类似的方程。回顾其定义：
$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]$
同样，代入 $G_t = R_{t+1} + \gamma G_{t+1}$：
$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a]$
$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[R_{t+1} | S_t = s, A_t = a] + \gamma \mathbb{E}_{\pi}[G_{t+1} | S_t = s, A_t = a]$
现在，期望是基于已经在状态 $s$ 中采取了动作 $a$。环境随后确定了根据 $p(s', r | s, a)$ 的后续状态 $s'$ 和奖励 $r$。

第一项是在状态 $s$ 中采取动作 $a$ 后的期望即时奖励：
$\mathbb{E}_{\pi}[R_{t+1} | S_t = s, A_t = a] = \sum_{s', r} p(s', r | s, a) r$
第二项涉及从后续状态 $S_{t+1} = s'$ 开始的期望折扣回报。由于智能体从状态 $s'$ 开始遵循策略 $\pi$，期望回报是 $\mathbb{E}_{\pi}[G_{t+1} | S_{t+1} = s'] = V^{\pi}(s')$。对所有可能的后续状态 $s'$ 和奖励 $r$ 进行平均：
$\gamma \mathbb{E}_{\pi}[G_{t+1} | S_t = s, A_t = a] = \gamma \sum_{s', r} p(s', r | s, a) V^{\pi}(s')$
将这些合并，得到 $Q^{\pi}$ 关于 $V^{\pi}$ 的贝尔曼期望方程：
$Q^{\pi}(s,a) = \sum_{s', r} p(s', r | s, a) [r + \gamma V^{\pi}(s')]$

这告诉我们，在状态 $s$ 中采取动作 $a$ 的值，是期望即时奖励加上*下一个状态*的折扣期望值，对所有可能的后续状态 $s'$ 和奖励 $r$ 进行平均。

> 状态-动作对 $Q^{\pi}(s,a)$ 的值是关于可能的后续状态 $s'$ 和奖励 $r$ 的期望值。每个产出 $(s', r)$ 以概率 $p(s', r | s, a)$ 发生，并将其即时奖励 $r$ 加上后续状态的折扣值 $\gamma V^{\pi}(s')$ 贡献给期望值。

我们也可以将 $Q^{\pi}(s,a)$ 完全用 $Q^{\pi}$ 表示，将 $V^{\pi}(s') = \sum_{a'} \pi(a'|s') Q^{\pi}(s',a')$ 代入方程：
$Q^{\pi}(s,a) = \sum_{s', r} p(s', r | s, a) \left[r + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s', a')\right]$

### 方程组

对于一个有限马尔可夫决策过程 (MDP)，$V^{\pi}$ 的贝尔曼期望方程给出了一个包含 $|S|$ 个未知数（所有 $s \in S$ 的值 $V^{\pi}(s)$）的 $|S|$ 个线性方程组。类似地，$Q^{\pi}$ 的方程给出了一个包含 $|S| \times |A|$ 个未知数（所有 $s \in S, a \in A$ 的值 $Q^{\pi}(s,a)$）的 $|S| \times |A|$ 个线性方程组。

这些方程非常核心，因为它们定义了一个一致性条件。状态（或状态-动作对）的值必须与其在策略 $\pi$ 下可能的后续状态所产生的期望值保持一致。如果我们知道策略 $\pi$ 和环境动力学（MDP），原则上，我们可以求解这个方程组来找到精确的值函数 $V^{\pi}$ 或 $Q^{\pi}$。这个过程被称为**策略评估**。

贝尔曼期望方程是动态规划（接下来会讲到）等方法评估策略和寻找最优策略的根本，前提是环境的完美模型可用。它们也促成了无模型方法，如时序差分学习（在后续章节中讨论），这些方法根据经验估计这些值。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 3
  一本广受认可的教材，全面介绍了强化学习概念，包括贝尔曼期望方程的详细推导和解释。
- [Dynamic Programming](https://books.google.com/books/about/Dynamic_Programming.html?id=oW22AAAAIAAJ) — Richard Bellman (1957)
  Publisher: Princeton University Press; Pages: xxv+342 pp.
  引入动态规划概念和贝尔曼方程的原创著作，这些方程构成了解决序列决策问题的数学基础。
- [Reinforcement Learning Lecture 3: Dynamic Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF3RuFi1AHhpWZA5fCg6zTNizMpieA0zmAjbEACMhBzY9cEjZ06uvNAko7KoqzFV2Yh7ikiVbrzR0uA9-H--hsknboVxO1_Htn6Zt6WQdRrA_Aj54XRV6uN_jiT9fa3FhwnXzkBRA==) — David Silver (2015)
  强化学习大学课程的讲义，在策略评估和动态规划的背景下，清晰简洁地解释了贝尔曼期望方程。
