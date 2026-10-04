---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "actor-critic-fundamentals"
sourceId: 3383
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/actor-critic-fundamentals"
title: "行动者-评论者架构基本原理"
description: "介绍行动者-评论者框架，其中策略（行动者）和价值函数（评论者）逼近器是独立的。"
order: 2
plots: []
sourceHash: "dd5cfd43fecbf79354e0600f3b7f90291a0a9df5f01f867897c829cf9cc8ee53"
sourceCorrections: []
---

标准策略梯度方法（如 REINFORCE）根据从整个回合中采样的回报 $G_t$ 来更新策略参数 (parameter) $\theta$。梯度估计通常表示为 $\nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(A_t|S_t) G_t]$。尽管是无偏的，但依赖完整的蒙特卡洛回报 $G_t$ 会引入较大的方差，因为回报取决于轨迹中所有后续的动作和奖励。单个过高或过低的奖励可能显著地影响梯度估计，从而减慢学习速度或使其不稳定。

行动者-评论者方法提供了一种有效的替代结构来降低这种高方差。学习策略时不再只有一个组件，我们引入了两个不同的组件，通常作为独立的函数逼近器（常为神经网络 (neural network)）实现：

1. **行动者 (The Actor)：** 该组件负责学习和表示策略。它接收当前状态 $s$ 作为输入，并输出动作的概率分布（对于随机策略）或一个特定动作（对于确定性策略）。我们将行动者的策略表示为 $\pi_\theta(a|s)$，其参数为 $\theta$。行动者的目标是通过调整 $\theta$ 来学习最优策略。
2. **评论者 (The Critic)：** 该组件学习一个价值函数，以评估行动者选择的动作或遇到的状态。它接收状态 $s$（有时也接收动作 $a$）作为输入，并输出一个价值估计。评论者函数常见的选择是状态价值函数 $V_\phi(s)$ 或动作价值函数 $Q_\phi(s,a)$，其参数为 $\phi$。评论者的作用是*不*选择动作，而是对行动者当前策略的好坏提供反馈。

### 它们如何协同工作

主要思想是行动者决定*做什么*，评论者评估*做得如何*。这种评估随后指导行动者的更新。典型的流程如下：

1. **动作选择：** 行动者观察当前状态 $S_t$，并根据其策略 $\pi_\theta(A_t|S_t)$ 选择一个动作 $A_t$。
2. **环境交互：** 动作 $A_t$ 在环境中执行，得到奖励 $R_{t+1}$ 和下一个状态 $S_{t+1}$。
3. **评论者评估：** 评论者使用转换信息 $(S_t, A_t, R_{t+1}, S_{t+1})$ 来评估行动者的动作或由此产生的状态。一种常见的方法是计算时序差分 (TD) 误差：
   
   $$
   \delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)
   $$
   
   此 TD 误差衡量评论者当前估计 $V_\phi(S_t)$ 与基于即时奖励和下一个状态的价值（TD 目标 $R_{t+1} + \gamma V_\phi(S_{t+1})$）的可能更好估计之间的差异。如果评论者估计 $Q_\phi(s,a)$，则 TD 误差可能是 $\delta_t = R_{t+1} + \gamma Q_\phi(S_{t+1}, A_{t+1}) - Q_\phi(S_t, A_t)$，其中 $A_{t+1}$ 是行动者在状态 $S_{t+1}$ 中选择的动作。
4. **评论者更新：** 评论者更新其参数 (parameter) $\phi$ 以最小化此 TD 误差，通常通过梯度下降 (gradient descent)。目标是随着时间推移使其价值估计更加准确。对于状态价值评论者，更新可能旨在最小化 $(\delta_t)^2$。
5. **行动者更新：** 行动者根据评论者的评估方向更新其策略参数 $\theta$。不再使用有噪声的蒙特卡洛回报 $G_t$，行动者使用评论者的反馈，通常是 TD 误差 $\delta_t$ 或相关的量，如优势 $A(S_t, A_t)$。策略梯度更新变为如下形式：
   
   $$
   \nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(A_t|S_t) \delta_t] \quad \text{(或使用优势)}
   $$
   
   正的 TD 误差表明动作 $A_t$ 导致了比预期更好的结果，因此，在状态 $S_t$ 中选择 $A_t$ 的概率应该增加。负的 TD 误差则表明相反。

这种交互循环使得行动者和评论者能够同时改进。评论者学习提供更好的评估，行动者学习根据这些评估产生更好的动作。

> 基本行动者-评论者交互循环。行动者根据状态选择动作，环境做出响应，评论者评估结果，提供反馈以更新行动者的策略及其自身的价值估计。

### 行动者-评论者结构的好处

与 REINFORCE 相比，行动者-评论者框架的主要好处是**方差降低**。用于行动者更新的 TD 误差 $\delta_t$（或优势）主要取决于即时奖励 $R_{t+1}$ 和评论者对下一个状态价值 $V_\phi(S_{t+1})$ 的估计。这种估计，虽然可能有偏差（因为 $V_\phi$ 是学习得到的），但通常比完整的蒙特卡洛回报 $G_t$ 噪声小得多，后者在许多时间步上累积噪声。较低的方差通常会带来更快、更稳定的收敛。

此外，行动者-评论者方法可以**在线**学习，在每一步（或一小批步）之后更新行动者和评论者，这与基本 REINFORCE 不同，后者通常需要等到回合结束才能计算 $G_t$。

### 变体和考量

评论者价值函数的具体形式对算法影响很大：

- **状态价值评论者 ($V_\phi(s)$)：** 评论者估计处于某个状态的价值。评估信号通常是 TD 误差 $\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$。在某些假设下，此 $\delta_t$ 直接逼近优势 $A(S_t, A_t)$。
- **动作价值评论者 ($Q_\phi(s,a)$)：** 评论者估计在状态 $s$ 中采取动作 $a$ 的价值。这在为连续动作空间（如 DDPG）设计的算法中很常见，或者当需要直接估计 Q 值时。更新通常涉及 TD 误差 $\delta_t = R_{t+1} + \gamma Q_\phi(S_{t+1}, A_{t+1}) - Q_\phi(S_t, A_t)$。
- **优势函数评论者：** 一些方法明确尝试估计优势函数 $A(s,a) = Q(s,a) - V(s)$。在策略梯度更新中直接使用优势，$\nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(A_t|S_t) A(S_t, A_t)]$，通常是有利的，因为它提供了一个相对衡量，表明某个动作相对于该状态的平均动作好多少，从而进一步降低方差。我们将在本章后面看到估计优势的复杂方法，如广义优势估计 (GAE)。

在深度强化学习 (reinforcement learning)中，行动者 $\pi_\theta$ 和评论者 ($V_\phi$ 或 $Q_\phi$) 通常都由神经网络 (neural network)表示。它们的参数 (parameter) $\theta$ 和 $\phi$ 是根据上述原理，使用基于梯度的优化技术进行更新的。

这种基本的行动者-评论者架构是许多强大算法的依据，我们接下来会讨论这些算法，包括优势行动者-评论者 (A2C/A3C)、深度确定性策略梯度 (DDPG)、近端策略优化 (PPO) 和软行动者-评论者 (SAC)。这些方法通过改进优势的估计方式、更新的执行方式以提高稳定性，或如何处理探索来完善基本结构。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本经典且全面的教科书，提供了强化学习的基础知识，包括对Actor-Critic方法的详细解释。
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: ICML 2016; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍了异步优势Actor-Critic（A3C）算法，这是一种极具影响力的深度强化学习方法，有效地将Actor-Critic范式与优势估计相结合。
- [Lecture Notes: Actor-Critic](https://web.stanford.edu/class/cs234/) — Emma Brunskill (2024)
  Journal: CS234: Reinforcement Learning; Publisher: Stanford University
  来自一所顶尖大学课程的全面讲义，提供了对Actor-Critic基础知识的清晰简洁解释，包括其推导和优点。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://doi.org/10.48550/arXiv.1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan and Pieter Abbeel (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  引入了广义优势估计（GAE），这是一种广泛采用的技术，可显著降低策略梯度方法的方差，对于高级Actor-Critic算法至关重要。
