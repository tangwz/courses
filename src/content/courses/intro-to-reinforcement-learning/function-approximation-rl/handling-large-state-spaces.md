---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "handling-large-state-spaces"
sourceId: 1650
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/handling-large-state-spaces"
title: "处理大规模状态空间"
description: "讨论为什么表格方法在具有庞大或连续状态空间的环境中会失效。"
order: 1
plots: ["plots/1650-0.json"]
sourceHash: "bccdc9c89ae1525e3ce8953315481e015b70ff6babd2a0e4f2e418c3bad68878"
sourceCorrections: []
---

Q学习和SARSA等算法通常通过构建表格来工作。对于Q学习，此表格存储每个可能的状态 $s$ 和动作 $a$ 的Q值 $Q(s, a)$。当状态和动作的数量相对较小时，例如在简单的Gridworld例子中，这种方法是有效的。每个条目的值可以根据经验直接存储和更新。

然而，强化学习 (reinforcement learning)的许多应用带来了一个重大的难题：状态空间的庞大规模。请看以下例子：

1. **游戏：** 像国际象棋这样的游戏估计有 $10^{47}$ 种可能的状态。围棋的状态数量更是天文数字，通常被认为比可观测宇宙中的原子数量（约 $10^{170}$）还要多。为每个状态存储Q表在物理上是不可能的。
2. **机器人技术：** 想象一个有多关节的机械臂。状态可以由每个关节的角度和速度来定义。由于这些是连续值，技术上存在无限多的状态。即使我们将角度和速度离散化，组合状态的数量也会随着关节数量的增加呈指数级增长（这通常被称为“维度灾难”）。例如，如果一个机器人有6个关节，并且我们将每个关节的角度离散化为仅100个位置，那么我们已经有 $100^6 = 10^{12}$ 个状态，这还没有考虑速度。
3. **资源管理：** 优化城市交通信号灯控制涉及表示大量交叉口交通流的状态，这些状态可以持续变化并取决于许多因素。

我们来设想一下状态空间增长的速度。如果状态的每个维度有 $D$ 个可能的值，并且我们有 $k$ 个维度，则状态总数为 $D^k$。



![状态空间规模的指数级增长](plots/1650-0.json)



> 状态数量 ($D^k$) 随状态维度数量 ($k$) 呈指数级增长，即使每维度 ($D$) 的值适中，也很快使表格方法在计算上变得不可行。请注意y轴的对数刻度。

这种爆发式增长给表格方法带来了几个问题：

- **内存需求：** 我们根本没有足够的内存（RAM或磁盘）来存储所需的庞大值表（V表或Q表）。
- **计算成本：** 即使我们可以存储表格，学习每个单一状态的值也需要大量的经验和计算时间。在实际训练期间，许多状态可能很少被访问，甚至从未被访问。
- **缺乏泛化能力：** 表格方法独立处理每个状态。学习一个状态并不会为其他可能非常相似的状态提供任何信息。如果机器人学习在位置 (1.0, 2.0) 处绕过障碍物，它没有关于如何在 (1.1, 2.0) 处绕过障碍物的固有知识，除非它专门体验并更新了其表格中的那个精确状态条目。

前进的方向是从为每个状态存储显式值转向**近似**值函数。我们需要一种能够将经验状态**泛化**到未见状态的机制。如果我们了解到某些状态是好是坏，我们希望推断出相似状态也可能是好是坏。

这就是**函数近似**的用武之地。我们不使用表格，而是使用一个带有一组参数 (parameter) $\mathbf{w}$ 的函数（如线性函数、神经网络 (neural network)或决策树）。此函数将状态 $s$（或状态-动作对 $(s, a)$）作为输入，并输出一个*估计*值：

- 状态值函数近似：$\hat{v}(s, \mathbf{w}) \approx V^\pi(s)$
- 动作值函数近似：$\hat{q}(s, a, \mathbf{w}) \approx Q^\pi(s, a)$

$\mathbf{w}$ 中的参数数量通常远小于状态总数 $|S|$。例如，我们可能有数十亿个状态，但我们的函数近似器中只需要几千或几百万个参数。通过根据经验调整这些参数 $\mathbf{w}$，我们旨在找到一个在整个状态空间（包括智能体之前未访问过的状态）中都能良好运行的近似。

接下来的部分将讨论如何以适合这些函数的方式表示状态，以及如何调整我们的学习算法（如TD学习）以更新参数 $\mathbf{w}$ 而不是表格条目。我们将从更简单的线性函数近似器开始，然后介绍更复杂的非线性近似器，例如神经网络。

## 参考资料

- [Reinforcement Learning: An Introduction](https://www.mitpress.mit.edu/books/reinforcement-learning-second-edition) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  强化学习的教材，提供了关于表格方法、大状态空间挑战以及函数逼近介绍的基础概念。
- [Dynamic Programming and Optimal Control, Vol. II: Approximate Dynamic Programming](http://athenasc.com/dpbook.html) — Dimitri P. Bertsekas (2012)
  Publisher: Athena Scientific
  一本全面的资源，详细介绍了近似动态规划方法，为在大型控制和强化学习问题中（特别是连续状态空间问题）使用函数逼近提供了理论基础。
