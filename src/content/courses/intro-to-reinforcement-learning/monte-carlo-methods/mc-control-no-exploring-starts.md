---
course: "intro-to-reinforcement-learning"
chapter: "monte-carlo-methods"
lesson: "mc-control-no-exploring-starts"
sourceId: 1635
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-4-monte-carlo-methods/mc-control-no-exploring-starts"
title: "无非固定起点MC控制"
description: "讨论类似$\\epsilon$-柔性策略的技术，以确保同策略MC控制中的试探。"
order: 5
plots: []
sourceHash: "835c0c73e8cb360881fb75cc446fbe052ef3745a2ba957749ace2ea64edf597f"
sourceCorrections: []
---

蒙特卡洛控制有时依赖于使用“非固定起点”。这意味着每个可能的（状态-动作）对 $(s, a)$ 都有非零概率成为一次情节的起始点。当情节从所有可能的起点开始时，可以保证智能体最终能在所有相关状态中针对所有动作获得经验（并计算到回报），从而使其能够学习最优动作值函数 $Q^*(s, a)$。

然而，非固定起点假定通常是不现实的。在许多模拟或物理系统中，我们没有条件从任意（状态-动作）对启动一次情节。比如训练一个机器人走迷宫：你可能可以将其重置到官方起始状态，但大概无法强制它在迷宫中间，刚转了一个特定弯道后开始。同样地，在游戏中，通常从预定义的初始状态开始。如果我们只从一个状态开始，并且总是遵循我们当前对策略的最佳猜测（即贪婪地行动），我们可能永远不会在某些状态下尝试某些动作，使潜在的改进未能被找到。我们可能会找到一个局部最优策略，却从未意识到存在一个更好的策略，而这个更好的策略只能通过我们从未尝试过的动作才能达到。

如果我们不能依赖非固定起点，如何才能确保足够的试探以找到最优策略呢？答案在于修改策略本身，以纳入试探行为。我们这里将侧重于*同策略*方法，在这些方法中，智能体学习的策略与它用于决策的策略是同一个。

### 确保使用$\epsilon$-柔性策略进行试探

核心思想很简单：智能体不应总是选择被认为最好的动作（即贪婪动作），而应该偶尔随机选择一个动作。这确保了，随着时间的推移，所有动作都能在所有被访问过的状态中被采样到。

在同策略学习中实现这一目标的常见方法是使用\*\*$\epsilon$-柔性策略\*\*。$\epsilon$-柔性策略指的是这样一种策略：它在给定状态 $s$ 下选择任意动作 $a$ 时，分配一个最小概率 $\frac{\epsilon}{|\mathcal{A}(s)|}$，并且 $|\mathcal{A}(s)|$ 是该状态下可用动作的数量，$\epsilon$ 是一个小的正数（例如 0.1）。

具体来说，对于一个$\epsilon$-柔性策略$\pi$：

- 当前根据动作值函数 $Q(s, a)$ 看起来最好的动作 $a^*$（即 $a^* = \arg\max_a Q(s, a)$）以 $1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$ 的概率被选择。
- 所有其他非贪婪动作以 $\frac{\epsilon}{|\mathcal{A}(s)|}$ 的概率被选择。

数学上，策略 $\pi(a|s)$ 定义为：


$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} & \text{如果 } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|\mathcal{A}(s)|} & \text{如果 } a \neq \arg\max_{a'} Q(s, a')
\end{cases}
$$


这保证了对于所有（状态-动作）对，$\pi(a|s) > 0$。参数 (parameter) $\epsilon$ 控制着试探（尝试随机动作）与利用（选择已知最佳动作）之间的权衡。较高的 $\epsilon$ 值表示更多的试探；较低的 $\epsilon$ 值表示更多的利用。

### 使用$\epsilon$-柔性策略的同策略MC控制

这如何融入我们遵循广义策略迭代（GPI）模式的蒙特卡洛控制循环呢？

1. **策略评估：** 我们使用当前$\epsilon$-柔性策略$\pi$来生成完整的经验情节。然后，通过平均在这些情节中每个（状态-动作）对 $(s, a)$ 之后观察到的回报（使用首次访问或每次访问MC预测），我们估计动作值函数 $Q^\pi(s, a)$。因为策略是$\epsilon$-柔性的，我们保证最终能够从任何被访问的状态中采样到所有动作。
2. **策略改进：** 在更新了对 $Q^\pi$ 的估计后，我们改进策略。我们使策略相对于*新的*动作值估计变为贪婪的。然而，为了保持试探，*改进的*策略 $\pi'$ 并非纯粹的贪婪策略，而是根据新的贪婪动作构建为$\epsilon$-柔性的。对于每个状态 $s$，找到动作 $a^* = \arg\max_a Q^\pi(s, a)$。新策略 $\pi'$ 则根据上述$\epsilon$-柔性定义设置，使用 $a^*$ 作为贪婪动作。

这种评估循环——使用$\epsilon$-柔性策略进行评估并向$\epsilon$-柔性贪婪策略改进——持续进行。这个过程将策略推向*最优$\epsilon$-柔性策略*。这种策略是在其必须以概率$\epsilon$进行试探的约束下最好的可能策略。它是$\epsilon$-柔性策略类别中最优的。

实践中，通常从较大的 $\epsilon$ 值开始（早期促使更多的试探），并随时间逐渐减小。如果 $\epsilon$ 衰减至零，策略将接近真正的最优策略 $\pi^*$。然而，确保收敛需要精心管理衰减计划以及理论条件（例如确保对所有（状态-动作）对的无限试探，通常总结为 GLIE——在无限试探下的极限贪婪）。

通过使用$\epsilon$-柔性策略，同策略蒙特卡洛方法能够有效地学习最优行为，即使非固定起点不可行，这使得它们适用于更广泛的问题，在这些问题中，学习必须完全通过持续的交互进行。下一节详细介绍使用此方法的同策略首次访问MC控制算法。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 5
  这是强化学习的权威教科书，全面涵盖了蒙特卡洛控制、在策略方法和epsilon-贪婪探索。
- [CS234: Reinforcement Learning (Winter 2024)](https://cs234.stanford.edu/) — Emma Brunskill (2025)
  Publisher: Stanford University; Pages: Lecture 4: Monte Carlo Methods
  提供强化学习大学课程的讲义和材料，内容涵盖蒙特卡洛控制和探索策略。
- [Reinforcement Learning (UCL Course)](https://www.davidsilver.uk/teaching/) — David Silver (2015)
  Pages: Lecture 5: Model-Free Control
  这个著名的系列讲座对强化学习概念提供了易于理解的解释，包括在策略蒙特卡洛控制和epsilon-贪婪策略。
