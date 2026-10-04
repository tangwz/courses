# Actor-Critic 方法概述

来源：[原文](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/actor-critic-methods-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管像 REINFORCE 这样的策略梯度方法允许我们直接优化策略，但它们在梯度估计中通常存在较大的方差。这是因为它们通常依赖于蒙特卡洛回报 $G_t$，即从时间步 $t$ 到回合结束的总累积奖励。回合后期单个高或低的奖励，会显著影响更早采取的动作的估计价值，使学习过程噪声大且缓慢。

Actor-Critic 方法提供了一种减小这种方差的方式，同时保留了策略梯度的优点。它们代表一种混合方式，结合了基于策略的方法（如 REINFORCE）和基于价值的方法（如 Q-learning 或 SARSA）的元素。

其核心结构中，一个 Actor-Critic 架构包含两个不同的组成部分，通常作为独立的神经网络 (neural network)或函数逼近器实现：

1. **行动者：** 该组成部分负责学习和执行策略。它接收当前状态 $s$ 作为输入，并输出动作的概率分布（对于离散动作空间）或分布的参数 (parameter)（对于连续动作空间）。行动者本质上是我们想要优化的参数化策略 $\pi_\theta(a|s)$。其目标是学习使期望回报最大化的参数 $\theta$。
2. **评论者：** 该组成部分估计一个价值函数，通常是状态价值函数 $V(s)$ 或动作价值函数 $Q(s,a)$。它接收状态信息（可能还有动作信息）作为输入，并输出处于该状态或在该状态下采取该动作的价值估计。评论者的作用是评估行动者所采取的动作，就其质量提供反馈。它学习自己的一组参数，通常表示为 $w$，用于价值函数逼近器（例如，$V_w(s)$ 或 $Q_w(s,a)$）。

### 它们如何一同工作

行动者和评论者同步学习，与环境和彼此进行交互。一个典型的交互循环如下所示：

1. **行动者执行动作：** 给定当前状态 $S_t$，行动者根据其当前策略 $\pi_\theta(A_t|S_t)$ 选择动作 $A_t$。
2. **环境响应：** 智能体在环境中执行动作 $A_t$，接收到下一个状态 $S_{t+1}$ 和奖励 $R_{t+1}$。
3. **评论者评估：** 评论者使用观察到的转移 $(S_t, A_t, R_{t+1}, S_{t+1})$ 来评估行动者的动作。一种常见的方式是计算时间差分 (TD) 误差，它衡量估计价值与观察到的奖励加上下一个状态的折扣价值之间的差值：
   - 如果使用状态价值函数 $V_w(s)$：
     
     $$
     \delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)
     $$
     
   - 如果使用动作价值函数 $Q_w(s, a)$：
     
     $$
     \delta_t = R_{t+1} + \gamma Q_w(S_{t+1}, A_{t+1}) - Q_w(S_t, A_t)
     $$
     
     ($A_{t+1}$ 的选择依据行动者的策略，或根据评论者的 Q 值导出，具体取决于特定算法的变体)。
     TD 误差 $\delta_t$ 提供了一个信号，表明在状态 $S_t$ 中采取动作 $A_t$ 的结果是否比评论者当前价值估计所预期的要好 ($\delta_t > 0$) 或差 ($\delta_t < 0$)。
4. **更新：**
   - **评论者更新：** 评论者更新其参数 (parameter) $w$ 以提高其价值估计，通常使用 TD 学习规则来最小化 TD 误差。对于 $V_w(S_t)$，更新可能看起来像 $w \leftarrow w + \alpha_C \delta_t \nabla_w V_w(S_t)$，其中 $\alpha_C$ 是评论者的学习率。
   - **行动者更新：** 行动者根据评论者的评估更新其策略参数 $\theta$。更新方向旨在增加导致好于预期结果（正 TD 误差）的动作的概率，并降低导致差于预期结果（负 TD 误差）的动作的概率。更新通常采取与策略梯度定理相关联的形式，但使用 TD 误差 $\delta_t$ 而不是完整的总回报 $G_t$：
     
     $$
     \theta \leftarrow \theta + \alpha_A \delta_t \nabla_\theta \log \pi_\theta(A_t|S_t)
     $$
     
     这里，$\alpha_A$ 是行动者的学习率。请注意评论者的评估 $\delta_t$ 如何直接调节策略更新。

> Actor-Critic 架构中的基本交互循环。行动者选择动作，评论者使用来自环境的信息评估它们，两个组成部分都根据此评估更新其参数。

### 相较于 REINFORCE 的优点

Actor-Critic 方法相较于像 REINFORCE 这样的基本策略梯度方法，其主要优点是**方差减小**。通过使用 TD 误差 $\delta_t$（其仅取决于下一个奖励 $R_{t+1}$ 和下一个状态 $S_{t+1}$ 的估计价值）而不是完整的蒙特卡洛回报 $G_t$，更新变得噪声较小。评论者提供了一个更稳定的、在训练期间自适应的学习基准。这通常会带来更快、更稳定的收敛。

此外，由于它们依赖于 TD 更新，Actor-Critic 方法可以进行**在线学习**（每一步之后更新），并且可以更自然地应用于**持续性任务**（没有明确结束的任务）。

Actor-Critic 方法是许多高级强化学习 (reinforcement learning)算法的根本。本次概述介绍了基本思想，存在多种变体，它们在价值函数估计方式、更新执行方式以及行动者和评论者网络结构上有所不同（例如，优势Actor-Critic (A2C)，异步优势Actor-Critic (A3C)，深度确定性策略梯度 (DDPG)）。然而，核心思想保持不变：使用评论者学习一个价值函数，该函数提供低方差反馈以指导行动者的策略更新。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  这本教科书是对强化学习的权威介绍，全面涵盖了包括REINFORCE在内的策略梯度方法，并详细解释了Actor-Critic算法及其优势。
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: ICML 2016; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  这篇论文介绍了A3C，一种影响深远的深度强化学习算法，它利用异步并行智能体高效训练Actor-Critic模型，展示了一种实用且可扩展的方法。
- [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — Vijay R. Konda, John N. Tsitsiklis (1999)
  Journal: Advances in Neural Information Processing Systems; Publisher: MIT Press; Volume: 12; Pages: 1008-1014
  一篇基础论文，提供了Actor-Critic算法的早期理论处理，通过形式化其结构和收敛特性，为后续发展奠定了基础。
- [Reinforcement Learning Lecture 6: Policy Gradient Methods](https://github.com/zyxue/youtube_RL_course_by_David_Silver/blob/master/L7-policy-gradient-methods.pdf) — David Silver (2015)
  Publisher: University College London (UCL)
  这份来自知名强化学习课程的讲座提供了策略梯度方法的出色概念性概述，并介绍了Actor-Critic作为减少方差和稳定学习的关键技术。

---

[上一节](04-%E9%99%8D%E4%BD%8E%E6%96%B9%E5%B7%AE%E7%9A%84%E5%9F%BA%E7%BA%BF.md) · [下一节](06-%E5%AF%B9%E6%AF%94%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E5%92%8C%E5%9F%BA%E4%BA%8E%E7%AD%96%E7%95%A5%E7%9A%84%E6%96%B9%E6%B3%95.md)
