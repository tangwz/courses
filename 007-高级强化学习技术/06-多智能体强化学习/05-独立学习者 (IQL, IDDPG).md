# 独立学习者 (IQL, IDDPG)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/independent-learners-marl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在多智能体情境中应用强化学习 (reinforcement learning)最直接的方式是简单地让每个智能体独立学习，将所有其他智能体视为环境动态的一部分。这种做法通常被称为**独立学习**。它不加修改地使用标准的单智能体强化学习算法，因此简单且易于实现。

### 主要想法

设想在一个环境中有 $N$ 个智能体。通过独立学习，每个智能体 $i \in \{1, ..., N\}$ 维护自己的策略 $\pi_i$ 或值函数（例如，Q函数 $Q_i$），并纯粹根据自身的经验进行学习。智能体 $i$ 的经验元组通常包含其局部观测 $s_i$、其采取的动作 $a_i$、其获得的奖励 $r_i$ 以及其下一个局部观测 $s'_i$。

在训练期间，智能体 $i$ 使用标准算法更新其策略或值函数，完全不顾及其他智能体 $(j \neq i)$ 也在学习并调整其策略 $\pi_j$ 这一事实。

### 常见实现：IQL 和 IDDPG

这种想法的两种常见实现是独立Q学习 (IQL) 和独立深度确定性策略梯度 (IDDPG)。

#### 独立Q学习 (IQL)

在IQL中，每个智能体 $i$ 独立地学习其动作值函数 $Q_i(s_i, a_i)$，使用标准的Q学习更新规则或其深度学习 (deep learning)变体DQN。如果使用表格Q学习，智能体 $i$ 的更新规则将是：


$$
Q_i(s_i, a_i) \leftarrow Q_i(s_i, a_i) + \alpha \left[ r_i + \gamma \max_{a'} Q_i(s'_i, a') - Q_i(s_i, a_i) \right]
$$


这里，$s_i$ 和 $s'_i$ 分别是智能体 $i$ 的当前和下一个状态（或观测），$a_i$ 是其动作，$r_i$ 是其奖励，$\alpha$ 是学习率，$\gamma$ 是折扣因子。如果使用神经网络 (neural network)（独立DQN），每个智能体将拥有自己的DQN网络，使用填充了其 $(s_i, a_i, r_i, s'_i)$ 转移的经验回放缓冲区进行训练。损失函数 (loss function)最小化目标Q值和预测Q值之间的差异，这与单智能体DQN中一样。

#### 独立深度确定性策略梯度 (IDDPG)

对于具有连续动作空间的环境，每个智能体都可以独立应用DDPG算法。在IDDPG中，每个智能体 $i$ 维护自己的执行者网络 $\mu_i(s_i | \theta^{\mu_i})$ 和评论者网络 $Q_i(s_i, a_i | \theta^{Q_i})$。每个智能体根据自身经验，使用标准的DDPG更新来更新其网络。执行者旨在最大化其评论者预测的预期回报，而评论者则学习精确估计执行者策略的Q值。同样，每个智能体在更新步骤中将其他智能体视为环境的静态部分。

> 每个智能体独立学习，与环境交互并接收自己的观测和奖励。智能体之间没有直接的学习更新或策略信息共享。

### 挑战：非平稳性问题

独立学习的简单性带来了显著的代价。正如章节引言中提到的，多智能体强化学习 (reinforcement learning)中的主要问题是非平稳性。从任何单个智能体 $i$ 的角度来看，环境似乎是非平稳的，因为其他智能体 $j \neq i$ 正在同时更新其策略。

考虑智能体 $i$ 的Q学习更新。目标值 $r_i + \gamma \max_{a'} Q_i(s'_i, a')$ 取决于下一个状态 $s'_i$。然而，$s'_i$ 不仅由智能体 $i$ 的动作 $a_i$ 决定，还由所有其他智能体 $j$ 所采取的动作 $a_j$ 决定。由于生成这些动作 $a_j$ 的策略 $\pi_j$ 正在改变，从智能体 $i$ 的角度来看，转移动态 $P(s'_i | s_i, a_i)$ 实际上也在变化。

这违反了Q学习和许多其他强化学习算法所依据的基本马尔可夫假设。该假设是环境的动态是平稳的（固定的）。当这个假设被破坏时：

1. **收敛问题：** Q学习等算法的理论收敛保证不再成立。学习过程可能出现振荡、发散或收敛到糟糕的策略。
2. **目标不稳定：** 更新中使用的目标值（如Q学习中的贝尔曼目标或DDPG中的评论者目标）变得不稳定，使学习变得困难。现在看起来好的动作，可能仅仅因为其他智能体改变了行为，在稍后就变得不好了。
3. **难以协调：** 由于智能体忽略彼此的学习过程，因此很难实现复杂的协调行为。智能体可能学会相互冲突的策略，或者未能收敛到互利的联合动作。

### 优缺点概述

**优点：**

- **简单：** 直接重用现有的单智能体强化学习 (reinforcement learning)算法。易于理解和实现。
- **较低的通信需求（执行时）：** 一旦训练完成，智能体通常只需要局部观测即可行动。

**缺点：**

- **非平稳性：** 核心问题，导致学习不稳定和潜在的糟糕表现。
- **缺乏协调：** 智能体不明确建模或考虑其他智能体，阻碍了复杂的合作或竞争。
- **适用性有限：** 在需要紧密协调或智能体之间结果相互影响较大的任务中，通常表现不佳。

### 何时使用独立学习者

尽管非平稳性是一个显著缺点，但独立学习并非一无是处。它在某些情况下可能有效：

- **简单环境：** 在智能体交互很少或最优策略不需要复杂协调的任务中。
- **基准：** 它可作为衡量更高级多智能体强化学习 (reinforcement learning)算法性能提升的基本参照。如果一个复杂方法无法超越IQL或IDDPG，则可能表明该方法存在问题，或者该问题不需要复杂的多智能体推理 (inference)。
- **资源限制：** 当实现复杂性或计算资源有限时。

然而，对于大多数复杂的多智能体问题，独立学习引入的非平稳性需要更专业的方法。通常需要明确处理智能体交互的技术，例如参数 (parameter)共享、集中训练与分散执行（CTDE）、值分解方法或多智能体策略梯度，才能实现稳定有效的学习。这些方法将在后续章节中介绍。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  本文介绍了深度Q网络（DQN），该单智能体算法是应用于高维观测空间时独立深度Q学习（IDQN）的基础。
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) — Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971)
  本文介绍了深度确定性策略梯度（DDPG），这是一种用于连续动作空间的无模型、离策略算法，它是独立DDPG（IDDPG）的底层方法。
- [A Survey of Learning in Multiagent Environments: Prioritization of Challenges and Solutions](https://link.springer.com/article/10.1007/s10462-018-9635-y) — Pablo Hernandez-Leal, Matthew Kaisers, Enrique Munoz de Cote (2019)
  Journal: Artificial Intelligence Review; Publisher: Springer; Volume: 52; Pages: 1675-1718; DOI: [10.1007/s10462-018-9635-y](https://doi.org/10.1007/s10462-018-9635-y)
  本调查概述了各种多智能体强化学习方法，包括独立学习，并仔细考察了多智能体环境中非平稳性问题。

---

[上一节](04-%E6%8C%91%E6%88%98%EF%BC%9A%E9%9D%9E%E9%9D%99%E6%80%81%E6%80%A7%E9%97%AE%E9%A2%98.md) · [下一节](06-%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB%E6%96%B9%E6%B3%95.md)
