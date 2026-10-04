---
course: "intro-to-reinforcement-learning"
chapter: "temporal-difference-learning"
lesson: "expected-sarsa"
sourceId: 1648
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/expected-sarsa"
title: "期望SARSA"
description: "介绍期望SARSA作为一种替代的TD控制算法。"
order: 7
plots: []
sourceHash: "0c23797eec412310e262f4da93235aa804a39ba6c67396d58cfd22daae1037e2"
sourceCorrections: []
---

强化学习 (reinforcement learning)算法以多种方式进行策略更新。例如，SARSA是一种同策略方法，根据当前策略$\pi$所执行的*实际*下一个动作($A'$)进行学习。相比之下，Q-learning是一种异策略方法，根据*贪婪*的下一个动作($\max_{a'} Q(S', a')$)进行学习。**期望SARSA**是另一种相关的同策略算法，它提供了与这些方法略有不同的变体，并可能具有一些有利的特性。

回顾SARSA的更新规则：


$$
Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]
$$


项$Q(S', A')$取决于在状态$S'$下从策略$\pi$中抽样得到的特定动作$A'$。这种抽样会给更新目标引入随机性（方差）。如果策略是高度随机的，或者某些动作的Q值明显不同，这种方差可能会减慢或影响学习的稳定性。

期望SARSA的目标是通过将单个抽样的下一个动作值$Q(S', A')$替换为下一个动作值的*期望*值来减小这种方差，该期望值是根据当前策略$\pi(a'|S')$对所有可能的下一个动作$a'$进行平均得到的。

期望SARSA的更新规则是：


$$
Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \sum_{a'} \pi(a'|S') Q(S', a') - Q(S, A) \right]
$$


让我们分解目标值$R + \gamma \sum_{a'} \pi(a'|S') Q(S', a')$：

- $R$：在状态$S$中执行动作$A$后立即获得的奖励。
- $\gamma$：折扣因子。
- $\sum_{a'} \pi(a'|S') Q(S', a')$：这是下一状态$S'$的期望Q值，通过将每个可能的下一个动作$a'$的Q值乘以在当前策略$\pi$下执行该动作的概率，然后求和计算得出。

本质上，期望SARSA不是等待查看策略接下来选择了哪个特定动作$A'$并使用其Q值，而是考虑*所有*可能的下一个动作及其*根据当前策略*的概率，从而计算出一个更平滑、平均化的目标。

### 与SARSA和Q-Learning的比较

期望SARSA与SARSA和Q-learning都有相似之处，但具有不同的特点：

- \*\*对比SARSA：\*\*两者都是同策略方法（更新使用了从当前遵循的策略$\pi$派生出的值）。然而，期望SARSA计算的是对下一个动作的期望，而SARSA则使用单个样本$A'$。这通常使期望SARSA具有较低的方差，从而能够带来更稳定的性能，尤其是在随机性高或接近“悬崖”的环境中，因为单个错误的样本可能具有误导性。如果动作空间很大，期望SARSA每次更新的计算成本会略高，因为它需要遍历所有可能的下一个动作$a'$。
- \*\*对比Q-Learning：\*\*更新规则看起来相似，但Q-learning在其目标中使用了$\max$运算符（$R + \gamma \max_{a'} Q(S', a')$），使其成为离策略方法，因为它总是朝着贪婪动作进行更新，而不论当前遵循的策略是什么。期望SARSA则根据当前策略$\pi$使用期望值，使其保持同策略。有趣的是，如果策略$\pi$相对于当前Q值变为贪婪策略（即当$a' = \arg\max_{a''} Q(S', a'')$时$\pi(a'|S') = 1$，否则为0），那么期望SARSA的更新将与Q-learning的更新完全相同。

以下是一个快速比较：

| 特点 | SARSA | Q-Learning | 期望SARSA |
| --- | --- | --- | --- |
| **更新目标** | $R + \gamma Q(S', A')$ | $R + \gamma \max_{a'} Q(S', a')$ | \$R + \gamma \sum\_{a'} \pi(a' |
| **策略类型** | 同策略 | 离策略 | 同策略 |
| **基于$A'$的依据** | 根据$\pi$*实际采取*的动作 | 在$S'$中产生*最大*Q值的动作 | 基于$S'$中$\pi$对所有动作$a'$的*期望值* |
| **方差** | 较高（使用采样$A'$） | 较低（使用最大值） | 通常低于SARSA（使用期望值） |

期望SARSA通常提供了一个很好的平衡，它保持了SARSA的同策略特性，同时实现了类似于Q-learning的方差减小，有时会带来改进的性能稳定性。它是TD控制算法系列中一个值得考虑的替代方案。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 6
  一本基础教材，全面解释了时序差分学习，包括SARSA、Q-学习和预期SARSA，涵盖了它们的特性和比较分析。
- [Reinforcement Learning Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2c_E408) — David Silver (2015)
  一位知名研究员的具影响力的讲座，在无模型控制的背景下，清晰地解释和比较了SARSA、Q-学习和预期SARSA。
