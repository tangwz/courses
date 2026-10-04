---
course: "intermediate-reinforcement-learning"
chapter: "policy-gradient-methods"
lesson: "value-based-methods-limitations"
sourceId: 5134
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/value-based-methods-limitations"
title: "基于价值方法的局限性"
description: "基于价值的方法（如DQN）遇到困难的情况（例如，连续动作空间）。"
order: 1
plots: []
sourceHash: "c64eeb472e1492277d7a5f1c9cab553fb09b0499e6fdd4db8a53dacddd73a2a6"
sourceCorrections: []
---

基于价值的强化学习 (reinforcement learning)算法，包括Q学习及其深度学习 (deep learning)扩展DQN，通过首先学习一个动作价值函数，$Q(s, a)$ 来运行。这个函数估算在状态 $s$ 下采取动作 $a$ 之后，并依循特定策略所能获得的预期回报。策略本身通常是从这些Q值隐式地得出的，通常是通过选择预期价值最高的动作（贪婪策略），或者使用像epsilon-greedy这样的变体来进行试探。

尽管功能强大，特别是在离散动作空间的环境中，这种基于价值的方法在某些情况下会遇到明显的困难。理解这些局限性有助于阐明为什么直接学习策略，正如我们将在策略梯度方法中操作的那样，会更有利。

### 处理连续动作空间的挑战

一个主要挑战出现在处理动作连续的环境中。设想控制一个机器人手臂，动作可能是对关节施加的精确扭矩，或者调整车辆的油门。这里的动作是某一范围内的实数，这意味着有无限多个可能的动作。

基于价值的方法，如DQN，依赖于寻找能使给定状态下的Q值最大化的动作：


$$
a^* = \arg\max_a Q(s, a)
$$


在离散动作空间中，这种最大化操作很简单：我们只需计算每个可能动作（有限集合）的Q值，然后选择对应最高Q值的动作。但是，如果动作空间 $a$ 是连续的（例如，$a \in \mathbb{R}$），那么在智能体需要采取动作的*每一步*执行这个最大化步骤，都会变成一个不简单的优化问题。在智能体的决策循环内部运行优化程序会带来高昂的计算成本并降低速度。

一个常见的权宜之计是将连续动作空间离散化。例如，与其允许-1.0到1.0之间的任何扭矩值，我们可能只允许{-1.0, -0.5, 0.0, 0.5, 1.0}这样的动作。虽然这使得 $\arg\max$ 操作再次可行，但它也带来了一系列自身的问题：

- **精度损失：** 最优动作可能位于离散点之间，从而限制智能体的表现。
- **维度灾难：** 如果动作空间有多个维度（例如，控制多个关节），离散动作组合的数量会呈指数级增长，使得计算所有Q值变得不切实际。

策略梯度方法，正如我们将看到的那样，通过直接输出动作概率分布的参数 (parameter)（例如，高斯分布的均值和标准差），或直接输出连续动作本身，更自然地处理连续动作空间。

### 表示最优随机策略的困难

基于价值的方法通常导致确定性策略。一旦Q值收敛，贪婪策略会在给定状态下选择具有最高Q值的单个动作。虽然epsilon-greedy在试探期间引入了一些随机性，但从Q函数推导出的*已学习*策略通常是确定性的。

然而，在某些环境中，最优策略本身是随机的。一个经典例子是剪刀石头布游戏。玩确定性策略（例如，总是出石头）会使你变得可预测且容易被利用。最优策略是概率性的：以相等概率（各1/3）出石头、布和剪刀。

考虑一个简单的网格场景，其中两个状态对智能体来说看起来完全相同（别名状态），但需要不同的动作才能有效率地达成目标。如果智能体学习一个确定性策略，它可能会陷入困境或表现不佳，因为它总是在两个视觉上相同的状态中采取相同的动作。一个最优随机策略可能涉及在这些别名状态中随机选择两个动作，从而随着时间的推移增加成功的机会。

基于价值的方法很难直接学习这类随机策略。尽管存在从Q值推导出随机策略的技术，但这通常不如那些显式地参数 (parameter)化和优化随机策略 $\pi(a|s; \theta)$ 的方法来得自然。

### 策略的简洁性与价值函数的复杂性

在某些问题中，最优策略本身可能比其对应的价值函数更容易表示。设想一个智能体需要平衡一根杆子。策略可能相对简单：“如果杆子向左倾斜，就向左推；如果杆子向右倾斜，就向右推。”然而，处于特定状态（杆子角度、角速度、小车位置、小车速度）的*价值*可能很难准确估算，它会严重依赖于该策略下的未来状态和动作。

学习价值函数 $V(s)$ 或 $Q(s, a)$ 的复杂细节可能需要一个非常复杂的函数逼近器（一个大型神经网络 (neural network)）和大量的训练，即使最优行为由更简单的规则决定。在这种情况下，直接学习策略 $\pi(a|s; \theta)$ 的参数 (parameter) $\theta$ 可能更有效率，并且需要一个更简单的模型。策略梯度方法提供了一种直接优化策略参数的方式，可能可以避免首先学习一个复杂的价值函数。

这些局限性表明，基于价值的方法可能不是最有效或最高效的途径。这促使我们研究策略梯度方法，它们直接优化策略，为解决更广泛的强化学习 (reinforcement learning)问题提供了有力的替代方案，特别是那些涉及连续动作或需要随机策略的问题。我们现在将继续了解这些技术的理论原理和实际实现。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本基础教材，涵盖了基于价值和策略梯度的强化学习方法，阐述了它们的原理以及策略梯度方法在处理连续动作空间和学习随机策略等场景中的优势。
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) — Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971)
  介绍了深度确定性策略梯度（DDPG），一种专门为连续动作空间设计的算法，直接解决了传统基于价值方法（如DQN）的一个重要局限性。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  提出了近端策略优化（PPO），一种广泛使用且稳健的策略梯度算法，以其在各种连续和离散控制任务中的表现而闻名，展示了直接策略优化的有效性。
