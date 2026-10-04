---
course: "advanced-reinforcement-learning"
chapter: "rl-foundations-revisited"
lesson: "bellman-optimality-review"
sourceId: 3346
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited/bellman-optimality-review"
title: "贝尔曼方程与最优性条件"
description: "再看值函数和Q函数的贝尔曼期望方程和最优性方程。"
order: 2
plots: []
sourceHash: "0149abb4536e269946b31de7af5a62e3e3760277c8b762d980a1f603da0fe8d3"
sourceCorrections: []
---

为了评估马尔可夫决策过程（MDP）框架内不同策略的有效性或“优劣”，并最终找到最优策略，一个数学基础不可或缺。这种评估是强化学习 (reinforcement learning)的核心。贝尔曼方程提供了将状态（或状态-动作对）的值与后续状态的值关联起来的数学依据，许多强化学习算法都以此为构建依据。

### 贝尔曼期望方程：评估给定策略

假设我们有一个固定的策略 $\pi$，它将状态映射到采取每个可能动作的概率 $\pi(a|s)$。在遵循此策略的情况下，处于特定状态 $s$ 有多大价值？我们使用状态值函数 $V^\pi(s)$ 来量化 (quantization)这一点，它定义为从状态 $s$ 开始并随后遵循策略 $\pi$ 所获得的期望总折扣回报：

$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s \right]$

这里，$\mathbb{E}_\pi[\cdot]$ 表示在智能体遵循策略 $\pi$ 的情况下的期望值，$R_{t+k+1}$ 是在时间步 $t+k+1$ 收到的奖励，且 $\gamma \in [0, 1]$ 是用于权衡即时奖励与未来奖励的折扣因子。

贝尔曼期望方程为 $V^\pi(s)$ 提供了递归定义。它将状态的值分解为期望的即时奖励和*下一个*状态的期望折扣值：

$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma V^\pi(s')]$

让我们分析此方程：

- $\sum_{a} \pi(a|s)$：我们对智能体在状态 $s$ 中可能采取的所有动作 $a$ 进行求和，并按策略 $\pi$ 下采取该动作的概率进行加权。
- $\sum_{s', r} p(s', r|s, a)$：对于选定的动作 $a$，我们对所有可能的下一个状态 $s'$ 和奖励 $r$ 进行求和，并按环境的转移动态 $p(s', r|s, a)$ 进行加权。这给出了在状态 $s$ 中采取动作 $a$ 的期望结果。
- $[r + \gamma V^\pi(s')]$：这是递归的核心。它是即时奖励 $r$ 加上下一个状态 $s'$ 的折扣值 $V^\pi(s')$，假设我们从 $s'$ 开始继续遵循策略 $\pi$。

类似地，我们可以定义动作值函数 $Q^\pi(s, a)$，它表示从状态 $s$ 开始，采取动作 $a$，*然后*遵循策略 $\pi$ 的期望回报：

$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a \right]$

$Q^\pi(s, a)$ 的贝尔曼期望方程是：

$Q^\pi(s, a) = \sum_{s', r} p(s', r|s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$

该方程表明，在状态 $s$ 中采取动作 $a$ 的值是期望的即时奖励加上*下一个状态-动作对*的期望折扣值，在下一个状态 $s'$ 中，对所有根据策略 $\pi$ 选择的可能的下一个动作 $a'$ 进行平均。

注意 $V^\pi$ 和 $Q^\pi$ 之间的直接关系：

- 状态的值是根据策略，该状态中可用动作值的期望值：
  $V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s, a)$
- 状态-动作对的值是期望的即时奖励加上下一个状态的期望折扣值：
  $Q^\pi(s, a) = \sum_{s', r} p(s', r|s, a) [r + \gamma V^\pi(s')]$

对于给定的策略 $\pi$ 和已知的动态 $p(s', r|s, a)$，贝尔曼期望方程定义了一个线性方程组。如果状态空间足够小，我们可以直接求解此系统以找到与策略 $\pi$ 对应的 $V^\pi$ 或 $Q^\pi$ 值函数。

> $Q^\pi(s, a)$ 的递归计算。该值取决于期望的即时奖励 $r$ 和下一个状态-动作对 $Q^\pi(s', a')$ 的折扣期望值，这些值根据环境动态和策略在下一个状态中的选择进行平均。

### 贝尔曼最优性方程：寻找最佳策略

评估给定策略虽然有用，但许多强化学习 (reinforcement learning)设置中的最终目标是找到一个*最优*策略，记为 $\pi_*$，该策略能从任何起始状态获得尽可能高的期望回报。我们将最优状态值函数 $V^*(s)$ 和最优动作值函数 $Q^*(s, a)$ 定义为：

$V^*(s) = \max_{\pi} V^\pi(s)$
$Q^*(s, a) = \max_{\pi} Q^\pi(s, a)$

贝尔曼最优性方程定义了最优策略的值。与期望方程不同，它们包含一个最大化步骤，反映了最优策略必须在每个状态中选择最佳动作这一事实。

$V^*(s)$ 的贝尔曼最优性方程是：

$V^*(s) = \max_{a} \sum_{s', r} p(s', r|s, a) [r + \gamma V^*(s')]$

该方程表明，状态 $s$ 的最优值是通过选择动作 $a$ 来获得的，该动作最大化了即时奖励 $r$ 和所得下一个状态 $s'$ 的折扣最优值 $\gamma V^*(s')$ 的期望和。

$Q^*(s, a)$ 的贝尔曼最优性方程通常更便于决策：

$Q^*(s, a) = \sum_{s', r} p(s', r|s, a) [r + \gamma \max_{a'} Q^*(s', a')]$

在这里，在状态 $s$ 中采取动作 $a$ 的最优值是期望的即时奖励加上在下一个状态 $s'$ 中可用的*最佳可能动作* $a'$ 的期望折扣值。术语 $\max_{a'} Q^*(s', a')$ 等同于 $V^*(s')$，突出了这种联系：

$V^*(s) = \max_{a} Q^*(s, a)$

重要地，贝尔曼最优性方程由于 $\max$ 运算符的存在而形成一个*非线性*方程组。求解此系统会得到最优值函数 $V^*$ 和 $Q^*$。一旦我们有了 $Q^*$，寻找最优策略 $\pi_*$ 就很简单了：在任何状态 $s$ 中，根据 $Q^*$ 采取贪婪行动。也就是说，选择使 $Q^*(s, a)$ 最大化的动作 $a$：

$\pi_*(s) = \arg\max_{a} Q^*(s, a)$
（如果多个动作具有相同的最大值，则可以选择其中任何一个。）

这些最优性方程并没有直接告诉我们*如何*找到 $V^*$ 或 $Q^*$，但它们定义了最优值必须满足的条件。值迭代和策略迭代（将在下文讨论）等算法运用这些方程来计算模型 $p(s', r|s, a)$ 已知情况下的最优策略。对于免模型设置，Q学习等方法根据经验迭代地逼近 $Q^*$。因此，理解贝尔曼方程对于掌握几乎所有基本的强化学习算法都非常必要。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  强化学习的标准综合教材，涵盖马尔可夫决策过程、贝尔曼方程和动态规划算法等基础内容。
- [Dynamic Programming and Optimal Control, Vol. I](http://www.athenasc.com/dpbook.html) — Dimitri P. Bertsekas (2017)
  Publisher: Athena Scientific
  一本关于动态规划的严谨数学书籍，包含贝尔曼方程，对深入理解最优控制很有帮助。
- [Lecture 3: Dynamic Programming](https://web.stanford.edu/class/cs234/) — Emma Brunskill (2024)
  Journal: Stanford University CS234: Reinforcement Learning Lecture Notes; Publisher: Stanford University
  在强化学习背景下，提供了贝尔曼方程、值迭代和策略迭代的学术概述。
