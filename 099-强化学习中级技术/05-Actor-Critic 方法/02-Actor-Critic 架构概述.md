# Actor-Critic 架构概述

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/actor-critic-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Actor-Critic 方法旨在结合基于价值和基于策略方法的优点。它们通过维护两个独立的组成部分来实现这一点，这两个部分通常由独立的神经网络 (neural network)（或有时是带有共享层的网络）表示，并并发地学习和运行：Actor 和 Critic。

### Actor：策略制定者

Actor 的作用是控制智能体的行为方式。它直接学习并表示策略，该策略是从状态到动作（或动作的概率分布）的映射。可以将 Actor 看作是负责决定在给定情境下*做什么*的组成部分。

- **功能：** 学习参数 (parameter)化的策略，记作 $\pi_\theta(a|s)$，其中 $\theta$ 代表参数（例如，神经网络 (neural network)的权重 (weight)）。
- **输出：** 对于离散动作空间，Actor 通常输出每个可能动作的概率。对于连续动作空间，它可能输出概率分布的参数（例如高斯分布的均值和标准差），从中采样动作。
- **目标：** 调整其参数 $\theta$，使策略最大化预期的累积奖励。它通过“听取”Critic 提供的反馈来实现这一点。

### Critic：动作评估者

Critic 的作用是评估 Actor 所采取的动作。它本身不决定动作。相反，它学习一个价值函数，该函数估计处于某个状态有多好，或在该状态下采取特定动作有多好。可以将 Critic 看作是负责判断 Actor 所选动作*有多好*的组成部分。

- **功能：** 学习参数 (parameter)化的价值函数，通常是状态价值函数 $V_\phi(s)$ 或动作价值函数 $Q_\phi(s, a)$，其中 $\phi$ 代表 Critic 的参数。在许多现代 Actor-Critic 变体中（例如我们稍后会看到的 A2C），Critic 主要学习状态价值函数 $V_\phi(s)$。
- **输出：** 一个标量值，表示从当前状态或状态-动作对估计的回报（累积未来奖励）。
- **目标：** 准确估计 Actor 当前遵循的策略下的价值函数。它通过观察由 Actor 与环境交互产生的奖励和状态转移来学习这一点，通常使用与时序差分 (TD) 学习相关的方法。

### 交互循环

Actor-Critic 架构的威力源于这两个组成部分在学习过程中的交互。

1. **动作选择：** Actor 观察环境中的当前状态 $s_t$，并根据其当前策略 $\pi_\theta(a|s_t)$ 选择动作 $a_t$。
2. **环境交互：** 智能体在环境中执行动作 $a_t$，获得奖励 $r_{t+1}$ 并转移到下一个状态 $s_{t+1}$。
3. **Critic 评估：** Critic 观察转移 $(s_t, a_t, r_{t+1}, s_{t+1})$。它使用此信息，通常结合其对 $s_t$ 和 $s_{t+1}$ 的当前价值估计，来计算一个评估动作 $a_t$ 的信号。一个常见的评估信号是 TD 误差或优势函数 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$，它衡量所采取的动作是否导致比状态 $s_t$ 预期更好或更差的结果。
4. **参数 (parameter)更新：**
   - **Critic 更新：** Critic 更新其参数 $\phi$ 以改进其价值估计，基于观察到的奖励和下一个状态，通常通过最小化基于 TD 误差的损失（例如，最小化 $(r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2$）。
   - **Actor 更新：** Actor 根据 Critic 提供的评估信号更新其策略参数 $\theta$。如果 Critic 表明动作 $a_t$ 导致了比预期更好的结果（正向 TD 误差或优势），Actor 会调整 $\theta$ 以增加未来在状态 $s_t$ 中选择 $a_t$ 的概率。反之，如果结果比预期差，则降低该概率。此更新通常遵循策略梯度建议的方向，但按 Critic 的评估进行缩放。

这种交互使得 Actor 能够从 Critic 学到的价值函数中获益。不再仅仅依赖于整个回合中通常带有噪声的回报（如基本 REINFORCE 中那样），Actor 从 Critic 的 TD 误差或优势估计中获得更即时和稳定的反馈。这通常会降低策略更新的方差，并提高学习效率。

> 此图展示了典型 Actor-Critic 设置中的信息流。包含 Actor 和 Critic 的智能体与环境交互。Actor 根据状态选择动作，环境提供奖励和下一个状态，而 Critic 评估 Actor 的动作，提供用于更新这两个组成部分的反馈。

通过分离动作选择和动作评估的任务，Actor-Critic 方法提供了一个灵活而有力的框架。在接下来的部分中，我们将介绍具体的算法，例如优势 Actor-Critic (A2C) 和异步优势 Actor-Critic (A3C)，它们有效地实现了这些思想。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  涵盖强化学习基本概念的综合性教科书，包括价值函数、策略梯度和Actor-Critic方法。
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) — Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour (2000)
  Journal: Advances in Neural Information Processing Systems; Volume: 13; Pages: 1057-1063
  介绍了策略梯度定理，为Actor-Critic算法中更新Actor策略参数提供了理论基础。
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: International Conference on Machine Learning (ICML); Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  提出了A3C，这是一种重要的深度Actor-Critic算法，有效实现了描述的架构，并在多种环境中展现了其效率。

---

[上一节](01-%E7%BB%93%E5%90%88%E7%AD%96%E7%95%A5%E5%92%8C%E4%BB%B7%E5%80%BC%E8%AF%84%E4%BC%B0.md) · [下一节](03-%E4%BC%98%E5%8A%BF%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%20%28A2C%29.md)
