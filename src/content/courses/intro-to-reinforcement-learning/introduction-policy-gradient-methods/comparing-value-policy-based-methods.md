---
course: "intro-to-reinforcement-learning"
chapter: "introduction-policy-gradient-methods"
lesson: "comparing-value-policy-based-methods"
sourceId: 1670
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/comparing-value-policy-based-methods"
title: "对比基于价值和基于策略的方法"
description: "讨论基于价值（如Q-学习）和基于策略的方法的优缺点。"
order: 6
plots: []
sourceHash: "d4e8975f877238947267811f5c327c34c719f81cf5a2307a7565090364f173e5"
sourceCorrections: []
---

强化学习 (reinforcement learning)问题可以用根本不同的方式来解决。第4章和第5章介绍了**基于价值的方法**，如蒙特卡洛、SARSA和Q-学习，其中主要目标是学习价值函数的精确估计（无论是状态价值 $V(s)$ 还是动作价值 $Q(s,a)$）。一旦我们对最优动作价值函数 $Q^*(s,a)$ 有了良好的估计，推导出最优策略通常很直接，例如，通过对 $Q^*$ 采取贪心行动。

本章介绍了**基于策略的方法**，以REINFORCE为例。在这里，策略转变了：我们直接参数 (parameter)化策略本身，$\pi_\theta(a|s)$，并学习参数 $\theta$ 以使预期回报最大化，通常使用梯度上升。价值函数可能仍然会被估计（例如，作为基线），但它不是学习的主要目标；策略才是。

理解这两种算法家族之间的权衡对于为给定问题选择合适的方法很重要。让我们比较一下它们的特点。

### 动作空间的处理

- **基于价值的方法：** 这些方法通常最适合**离散动作空间**。寻找最佳动作涉及选择Q值最大的那个，通常通过 $argmax_a Q(s,a)$ 操作。将其直接应用于**连续动作空间**会遇到问题。在连续域中寻找函数的最大值可能计算成本高昂，或需要在每个决策步骤*内*执行特定的优化步骤。尽管存在调整基于价值方法的技术（如离散化动作空间或使用特定网络架构），但这通常不如策略梯度方法自然。
- **基于策略的方法：** 策略梯度相当自然地处理**连续动作空间**。参数 (parameter)化策略 $\pi_\theta(a|s)$ 可以直接输出动作概率分布的参数，而不是为每个动作输出一个价值。例如，在连续空间中，策略可能输出高斯分布的均值 $\mu_\theta(s)$ 和标准差 $\sigma_\theta(s)$，并从中采样动作 $a$。优化 $\theta$ 会调整分布，以偏向更高奖励的动作。

> 基于价值方法与基于策略方法选择动作时信息流的比较。基于策略的方法直接将状态映射到动作（概率），而基于价值的方法通常通过学习到的动作价值进行。

### 学习策略的性质

- **基于价值的方法：** 像Q-学习这样的标准基于价值方法通常收敛到**确定性最优策略**（忽略像$\epsilon$-贪心这样的探索策略，这些策略在学习期间使用，但不属于收敛后的最优策略本身）。如果多个动作具有相同的最大Q值，则可以任意或随机地打破平局，但从$Q^*$贪心推导出的潜在学习策略通常是确定性的。
- **基于策略的方法：** 这些方法可以明确地学习**随机策略**。策略网络 $\pi_\theta(a|s)$ 输出概率 $P(a|s, \theta)$。这在多种情况下具有优势：
  - **部分可观测环境：** 当智能体没有关于状态的完整信息时，有时最好的策略本质上是随机的，以处理状态混叠（即不同的底层状态对智能体来说看起来相同）。
  - **策略考量：** 在多智能体设置或游戏（如石头-剪刀-布）中，确定性策略容易被利用，而随机策略可能是最优的。
  - **简洁性：** 有时，一个好的随机策略比一个会导致相同行为的复杂价值函数更容易表示和学习。

### 学习稳定性和效率

- **基于价值的方法：** 像Q-学习和SARSA这样的算法使用**时间差分（TD）更新**从单个转换中学习。这种自举方法（基于其他估计来更新估计）通常导致**更高的样本效率**，意味着它们可以用相对较少的环境交互来学习好的策略，尤其是在离散域中。然而，将TD学习与离策略训练和函数近似（如DQN中的神经网络 (neural network)）结合，有时会导致训练期间的**不稳定**，需要经验回放和目标网络等技术来缓解。
- **基于策略的方法：** 像REINFORCE这样的基本策略梯度方法依赖于**蒙特卡洛更新**，这意味着它们只有在观察到完整回合的完整回报 $G_t$ 后才更新策略参数 (parameter) $\theta$。这常常导致梯度估计的**高方差**，因为回报依赖于一长串动作和状态转换。高方差会减缓学习速度或使其不稳定。使用**基线**（如前一节讨论的）或转向Actor-Critic方法等技术对于减少方差、提高稳定性和样本效率很重要。尽管策略梯度直接优化性能目标，有时可以带来更平滑的收敛，但高方差是一个重要的实际挑战。

### 比较总结

| 特点 | 基于价值的方法（例如，Q-学习） | 基于策略的方法（例如，REINFORCE） |
| --- | --- | --- |
| **主要目标** | 学习精确的价值函数 ($Q^*(s,a)$) | 学习最优策略参数 (parameter) ($\theta$) |
| **策略输出** | 通常是确定性的（从价值推导） | 可以是随机的 |
| **动作空间** | 最适合离散动作 | 自然处理连续动作 |
| **样本效率** | 通常更高（由于TD更新） | 可能更低（特别是蒙特卡洛版本） |
| **梯度方差** | 更低（TD误差） | 更高（蒙特卡洛回报） |
| **稳定性** | 函数近似时可能不稳定 | 可能因高方差而不稳定 |
| **离策略** | Q-学习天然是离策略的 | 离策略学习可能更复杂 |

### 选择合适的方法

那么，你应该选择哪种方法呢？

- 如果你的问题具有**离散动作空间**并且**样本效率**是主要考虑，那么像Q-学习（或用于更大状态空间的DQN）这样的基于价值的方法可能是一个好的起点。
- 如果你的问题涉及**连续动作空间**或需要本质上**随机的策略**，策略梯度方法通常更适合。
- 如果你遇到基本策略梯度的高方差问题，请考虑加入**基线**或研究**Actor-Critic方法**，这是一种混合方法，试图结合两者的优点。我们接下来会提到Actor-Critic方法。

现代强化学习 (reinforcement learning)通常涉及结合了基于价值和基于策略两种方法的复杂算法，以发挥各自的优势并弥补其不足。理解这里讨论的基本差异为掌握这些更高级的技术提供了坚实的基础。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本关于基于价值和基于策略强化学习方法的理论基础和算法的全面教材，包括它们的详细比较。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amazed van den Heuvel, Demis Hassabis, and Daan Wierstra (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇论文介绍了深度Q网络（DQN），展示了如何将基于价值的方法与深度学习结合，并讨论了诸如不稳定性等挑战及经验回放等解决方案。
- [Actor-Critic Algorithms](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQExtwk9CmZ8X511UhH5gjl_9la2jSFCBX0x8TVncIa3_p8hX5DiJ0Laf0w4lKqE0U-51ekhm7g-zWh7tBQGYux6dSgDb-YWTBKaIlwonBmOv8DAzCsi5Mh-dRGJSYvFz2w_8LoCXLnj4g9YWctPW8TaYJA=) — Vijay R. Konda, John N. Tsitsiklis (2000)
  Journal: Advances in Neural Information Processing Systems; Publisher: The MIT Press; Volume: 12; Pages: 1008-1014
  一篇关于Actor-Critic架构的基础论文，该架构结合了基于策略和基于价值方法，以提高学习稳定性和效率。
