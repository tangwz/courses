---
course: "intro-to-reinforcement-learning"
chapter: "estimating-value-functions"
lesson: "solving-bellman-equations-overview"
sourceId: 1610
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-3-estimating-value-functions/solving-bellman-equations-overview"
title: "求解贝尔曼方程 (概述)"
description: "概述求解贝尔曼方程的方法。"
order: 3
plots: []
sourceHash: "a5a87f75578d820d138e4d85a2018bb6a62c6c2d6c5f46a22b564377fa20ecb9"
sourceCorrections: []
---

贝尔曼期望方程和最优性方程提供了强大的递归关系，可以描述给定策略 $\pi$ 的值函数 ($V^{\pi}, Q^{\pi}$) 以及最优策略 $\pi^*$ 的最优值函数 ($V^*, Q^*$)。例如，$V^{\pi}$ 的贝尔曼期望方程是：


$$
V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V^{\pi}(s')]
$$


而 $V^*$ 的贝尔曼最优性方程是：


$$
V^{*}(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma V^{*}(s')]
$$


尽管这些方程定义了值函数，但它们并不能直接给出具体的数值。我们现在的目标是找到*求解*这些方程的方法，也就是说，我们希望计算出所有状态 $s$ 对应的 $V^{\pi}(s)$ 或 $V^*(s)$ 的实际数值。

考虑 $V^{\pi}$ 的贝尔曼期望方程。如果我们知道策略 $\pi$ 和环境动态 $p(s', r | s, a)$，这个方程就定义了一个线性方程组。对于一个有 $|S|$ 个状态的系统，有 $|S|$ 个方程（每个 $V^{\pi}(s)$ 对应一个）和 $|S|$ 个未知数（所有 $s \in S$ 的 $V^{\pi}(s)$ 值）。原则上，如果 $|S|$ 较小，我们可以直接使用标准线性代数技术（例如矩阵求逆）来求解这个系统。然而，在许多实际问题中，状态数量 $|S|$ 可能非常庞大（数百万甚至无限），这使得直接求解在计算上不可行（矩阵求逆的复杂度通常为 $O(|S|^3)$）。

贝尔曼最优性方程带来了另一个挑战：其中的 $\max$ 运算符使得系统变为非线性。标准的线性方程求解器不能直接用于求解 $V^*$。

因此，我们通常采用**迭代方法**。这些方法从对值函数的初始估计开始，然后根据贝尔曼方程重复应用更新，逐步调整估计值，直到它们收敛到真实值。

*当环境的完整模型已知时*，用于求解贝尔曼方程的主要算法系列统称为**动态规划 (DP)**。DP 方法运用贝尔曼方程的结构来找到精确解，前提是您对转移概率 $p(s', r | s, a)$ 和奖励有完美的了解。

我们将介绍两种主要的DP方法：

1. **策略迭代：** 这种方法在两个步骤之间交替进行：

   - *策略评估：* 给定一个策略 $\pi$，通过迭代应用贝尔曼期望方程直到收敛来计算其值函数 $V^{\pi}$。
   - *策略改进：* 通过相对于计算出的值函数 $V^{\pi}$ 采取贪婪行动来改进策略。
     这个过程重复进行，直到策略稳定，从而保证收敛到最优策略 $\pi^*$ 和最优值函数 $V^*$。
2. **值迭代：** 这种方法通过对所有状态的值估计迭代应用贝尔曼最优性方程的更新规则，直接找到最优值函数 $V^*$。它将策略评估和改进的方面合并到一个更新步骤中。一旦值函数收敛到 $V^*$，就可以轻松地提取出最优策略 $\pi^*$。

> 动态规划方法求解贝尔曼方程的迭代特性，需要MDP模型作为输入。

这些DP方法构成了求解MDP的理论基础。接下来的章节将提供策略迭代和值迭代的详细说明和算法。请记住它们的一个重要要求：它们需要环境动态的完整准确模型。这个假设在实际中通常不成立，这也是后续章节中讨论的无模型强化学习 (reinforcement learning)技术的原因。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习领域的奠基性教材，全面涵盖马尔可夫决策过程、贝尔曼方程以及策略迭代和价值迭代等动态规划算法。
- [Dynamic Programming and Optimal Control, Vol. I](https://www.athenasc.com/dynamic-programming-and-optimal-control.html) — Dimitri P. Bertsekas (2017)
  Publisher: Athena Scientific; Pages: 576
  动态规划领域严谨而全面的著作，为解决贝尔曼方程的理论基础提供了重要的参考。
- [CS234: Reinforcement Learning (Autumn 2023-2024)](https://cs234.stanford.edu/) — Emma Brunskill (2025)
  Publisher: Stanford University
  一门高级大学课程，涵盖强化学习的理论和算法基础，包含贝尔曼方程和动态规划方法的讲座和材料。
