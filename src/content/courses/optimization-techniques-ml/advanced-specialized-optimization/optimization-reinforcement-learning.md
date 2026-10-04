---
course: "optimization-techniques-ml"
chapter: "advanced-specialized-optimization"
lesson: "optimization-reinforcement-learning"
sourceId: 1328
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-7-advanced-specialized-optimization/optimization-reinforcement-learning"
title: "强化学习策略的优化"
description: "策略梯度和 Q 学习方法中使用的特定优化难题及算法。"
order: 6
plots: []
sourceHash: "46aa1717af70104dd6b8dd76e4cad6c035c161b91d047a51de948fd3f510fc5c"
sourceCorrections: []
---

强化学习 (reinforcement learning) (RL) 带来了一系列独特的优化难题。这些难题与监督学习 (supervised learning)和无监督学习 (unsupervised learning)模式中遇到的不同。尽管RL的目标仍然是优化——寻找使目标最大化或最小化的参数 (parameter)，但该目标通常是智能体与环境交互所获得的预期累计奖励。优化过程因延迟奖励、试探需求以及学习智能体策略提升时数据分布变化等因素而复杂化。

强化学习算法的两个主要类别，即策略梯度方法和基于值的方法（如 Q 学习），非常依赖优化技术，并且经常调整本课程早期介绍的方法。

### 直接优化策略：策略梯度方法

策略梯度方法旨在直接优化智能体策略 $\pi_\theta(a|s)$ 的参数 (parameter)，用 $\theta$ 表示。该策略定义了在状态 $s$ 下采取行动 $a$ 的概率。目标通常是最大化预期总折扣奖励，常写为 $J(\theta)$。

核心难题在于估计梯度 $\nabla_\theta J(\theta)$，因为目标函数依赖于策略本身在环境中生成的交互轨迹。策略梯度定理提供了一种前进的途径，为该梯度提供了一个表达式。一种常用形式是：

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]$

这里，$\tau$ 表示轨迹 $(s_0, a_0, r_0, s_1, a_1, r_1, ...)$，$\pi_\theta(a_t | s_t)$ 是在策略 $\theta$ 下在状态 $s_t$ 采取行动 $a_t$ 的概率，而 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时间步长 $t$ 开始的总折扣回报。期望 $\mathbb{E}_{\tau \sim \pi_\theta}$ 取自遵循策略 $\pi_\theta$ 生成的轨迹。

这种公式允许我们使用智能体交互的样本来估计梯度。最简单的算法 REINFORCE 计算此梯度的蒙特卡洛估计，并使用梯度上升更新策略参数：

$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) G_t$

其中 $\alpha$ 是学习率。

**优化方面的考虑：**

1. **高方差：** 通过对 $G_t$ 进行蒙特卡洛采样获得的梯度估计通常存在高方差问题，导致学习缓慢且不稳定。这是策略梯度方法中的一个主要障碍。
2. **方差减小：** 一种减轻高方差的标准技术是引入一个仅取决于状态的基线 $b(s_t)$。梯度估计变为：
   $\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t | s_t) (G_t - b(s_t))$
   基线的一个常见选择是价值函数 $V(s_t)$，它估计从状态 $s_t$ 开始的预期回报。项 $(G_t - V(s_t))$ 是 *优势* $A(s_t, a_t)$ 的估计，从而产生了 Actor-Critic 方法，其中“评论者”（critic）学习价值函数（或优势函数）以改进“行动者”（actor，即策略）使用的梯度估计。
3. **优化器选择：** 由于梯度估计存在噪声，通常使用标准随机梯度上升（如果最小化负目标则使用下降）。Adam 或 RMSprop（第 3 章）等自适应学习率方法在强化学习 (reinforcement learning)中特别受欢迎，因为它们有助于处理噪声梯度和变化的参数敏感性。
4. **稳定性和步长：** 步长过大可能会显著改变策略，从而可能导致性能崩溃。这促使产生了诸如信任区域策略优化 (TRPO) 和近端策略优化 (PPO) 之类的方法。这些算法限制了策略更新步长，确保新策略在行为方面（通常通过 KL 散度衡量）与旧策略的偏离不会太大。这与信任区域（第 2 章）和约束优化思想相关联。

### 优化价值函数：Q 学习

基于值的方法，特别是 Q 学习及其深度学习 (deep learning)变体深度 Q 网络 (DQN)，采取了不同的方法。它们旨在学习最优行动价值函数 $Q^*(s, a)$，该函数表示从状态 $s$ 开始，采取行动 $a$，然后遵循最优策略所能达到的最大预期回报。

在 DQN 中，使用带有参数 (parameter) $\phi$ 的神经网络 (neural network)来近似 $Q^*(s, a)$，表示为 $Q(s, a; \phi)$。优化问题变为寻找使当前 Q 值估计与目标 Q 值之间差异最小化的参数 $\phi$，该目标 Q 值通常源自贝尔曼方程。这通常被表述为最小化均方贝尔曼误差 (MSBE)：

$L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (y - Q(s, a; \phi))^2 \right]$

其中目标 $y = r + \gamma \max_{a'} Q(s', a'; \phi^{-})$。这里，$\mathcal{D}$ 是存储过去经验 $(s, a, r, s')$ 的经验回放缓冲区，$\gamma$ 是折扣因子，而 $\phi^{-}$ 表示一个单独的 *目标网络* 的参数，这些参数会定期用主网络的参数 $\phi$ 进行更新。

**优化方面的考虑：**

1. **监督学习 (supervised learning)类比：** 优化 Q 网络类似于一个监督回归问题，其中网络尝试预测目标值 $y$。标准梯度下降 (gradient descent)技术适用。
2. **稳定性：** 训练深度 Q 网络可能不稳定。两种常用技术可以提高稳定性：
   - **经验回放：** 将经验存储在缓冲区 $\mathcal{D}$ 中并从中采样小批量，打破了顺序观察数据中的时间相关性，使样本更独立同分布 (i.i.d.)，这有利于标准梯度下降方法。
   - **目标网络：** 使用固定的目标网络 $Q(s', a'; \phi^{-})$ 为 $Q(s, a; \phi)$ 的梯度更新提供了稳定的目标 $y$。如果没有它，目标会随 $\phi$ 的每次更新而变化，可能导致震荡或发散。
3. **优化器选择：** 与策略梯度方法类似，RMSprop 和 Adam 等优化器因其在深度学习任务中的有效性而经常用于训练 DQN。学习率调度也可能被采用。

### 强化学习 (reinforcement learning)优化与通用技术的关联

强化学习中使用的优化方法通常是本课程中讨论的算法的特定调整或应用：

- **梯度下降 (gradient descent)变体：** SGD、Adam、RMSprop 构成了更新策略参数 (parameter)和价值函数近似器的核心。
- **约束优化/信任区域：** TRPO 和 PPO 等方法明确包含约束或信任区域以稳定策略更新，与本章早些时候和第 2 章中讨论的技术有相似之处。
- **超参数 (hyperparameter)优化：** 寻找合适的学习率、折扣因子、试探参数、网络架构和优化器设置非常重要。贝叶斯优化（本章早些时候介绍）是高效调整这些超参数的强大工具。

本质上，尽管强化学习引入了与智能体-环境交互相关的独特目标和挑战，但更新模型参数的基本机制很大程度上依赖于为更广泛的机器学习 (machine learning)应用开发的高级优化技术。理解这些优化算法为处理复杂的强化学习问题提供了坚实的支撑。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  涵盖强化学习基本概念和算法的标准教科书，包括策略梯度方法和Q学习。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍深度Q网络（DQN），展示了如何通过经验回放和目标网络将Q学习与深度神经网络结合。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  提出PPO算法，通过限制策略更新来提高策略梯度方法的稳定性和性能。
