# 多智能体深度确定性策略梯度 (MADDPG)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/maddpg-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

独立学习方法将其他智能体视为环境的一部分，这导致非平稳性问题。价值分解方法常需要特定的协作结构。这些局限性凸显了对一种更通用方法的需求，这种方法能够处理混合协作-竞争情况及连续动作空间。多智能体深度确定性策略梯度 (MADDPG) 算法正是这样一种解决方案。

MADDPG 基于单智能体深度确定性策略梯度 (DDPG) 算法，并巧妙地采用了集中式训练分散式执行 (CTDE) 方法。它的核心思路是为每个智能体学习一个集中式评论器，该评论器考虑*所有*智能体的观察和动作，而每个智能体则维护自己的分散式执行者，该执行者仅使用局部观察进行执行。

### MADDPG 架构

想象 $N$ 个智能体在一个环境中进行交互。在 MADDPG 中：

1. **分散式执行者：** 每个智能体 $i$ 都有其自己的执行者网络 $\mu_i$，由参数 (parameter) $\theta_i$ 参数化。这个执行者以智能体的局部观察 $o_i$ 作为输入，并输出一个确定性动作 $a_i = \mu_i(o_i; \theta_i)$。这与 DDPG 中的执行者相同，并允许分散式执行，因为每个智能体只需自己的观察来决定其动作。
2. **集中式评论器：** 每个智能体 $i$ 也有一个对应的评论器网络 $Q_i$，由参数 $\phi_i$ 参数化。与只接受智能体自身状态和动作的 DDPG 评论器不同，MADDPG 评论器 $Q_i$ 接受*所有*智能体的联合观察（或状态）$\mathbf{x} = (o_1, o_2, ..., o_N)$ 和*所有*智能体采取的联合动作 $\mathbf{a} = (a_1, a_2, ..., a_N)$ 作为输入。它输出智能体 $i$ 的估计 Q 值：$Q_i(\mathbf{x}, \mathbf{a}; \phi_i)$。

这个集中式评论器是解决非平稳性问题的核心所在。因为 $Q_i$ 观察所有智能体的动作，所以即使其他智能体的策略 $\mu_j$ ($j \neq i$) 在训练期间发生变化，评论器的学习目标也保持稳定。评论器在*所有其他智能体正在做的事情*的背景下，学习智能体 $i$ 动作的价值。

> MADDPG 架构概述。每个智能体 $i$ 使用其局部观察 $o_i$ 通过其执行者 $\mu_i$ 生成动作 $a_i$。在训练期间，经验存储在回放缓冲区中。采样批次后，联合信息（所有观察 $\mathbf{x}$、所有动作 $\mathbf{a}$）被送入每个智能体的集中式评论器 $Q_i$。评论器 $Q_i$ 通过策略梯度用于训练执行者 $\mu_i$。

### 训练 MADDPG 智能体

训练遵循执行者-评论器模式，利用集中式评论器将 DDPG 的更新规则扩展到多智能体场景。为了稳定性，我们为执行者 ($\mu'_i$) 和评论器 ($Q'_i$) 都维护目标网络，并像 DDPG 中那样，使用 Polyak 平均缓慢更新它们。

#### 评论器更新

每个评论器 $Q_i$ 通过最小化一个标准均方贝尔曼误差 (MSBE) 损失进行更新。我们从共享回放缓冲区 $D$ 中采样一个转换小批量 $(\mathbf{x}, \mathbf{a}, \mathbf{r}, \mathbf{x}')$。这里，$\mathbf{x}'$ 表示联合的下一个观察，$\mathbf{r} = (r_1, ..., r_N)$ 则包含每个智能体的奖励。智能体 $i$ 的评论器的目标值 $y_i$ 使用*目标*网络计算：


$$
y_i = r_i + \gamma Q'_i(\mathbf{x}', a'_1, ..., a'_N; \phi'_i) \big|_{a'_j = \mu'_j(o'_j)}
$$


此处，下一个动作 $a'_j$ 是基于 $\mathbf{x}'$ 中的下一个局部观察 $o'_j$，使用目标执行者 $\mu'_j$ 计算的。评论器 $i$ 的损失函数 (loss function)如下：


$$
L(\phi_i) = \mathbb{E}_{(\mathbf{x}, \mathbf{a}, \mathbf{r}, \mathbf{x}') \sim D} \left[ (Q_i(\mathbf{x}, \mathbf{a}; \phi_i) - y_i)^2 \right]
$$


此损失通过梯度下降 (gradient descent)进行最小化。重要的一点是，$Q'_i$ 使用来自*所有*目标执行者 $\mu'_j$ 的动作，确保目标计算与目标策略预期的联合行为一致。

#### 执行者更新

每个执行者 $\mu_i$ 旨在产生能最大化其预期回报的动作，此回报由其对应的集中式评论器 $Q_i$ 估计。智能体 $i$ 的策略梯度推导方式与 DDPG 类似，但它使用多智能体评论器 $Q_i$：


$$
\nabla_{\theta_i} J(\theta_i) \approx \mathbb{E}_{\mathbf{x} \sim D, \mathbf{a} \sim \boldsymbol{\mu}} \left[ \nabla_{\theta_i} \mu_i(o_i) \nabla_{a_i} Q_i(\mathbf{x}, a_1, ..., a_N; \phi_i) \big|_{a_i = \mu_i(o_i)} \right]
$$


我们来分解一下这个梯度：

- 我们从回放缓冲区 $D$ 中采样联合观察 $\mathbf{x}$。
- 我们使用*所有*智能体的*当前*执行者计算它们的当前动作：$a_j = \mu_j(o_j; \theta_j)$。
- 我们评估评论器 $Q_i$ 相对于智能体 $i$ 动作 $a_i$ 的梯度。这表示在保持其他智能体动作在此梯度步骤中不变的情况下，改变智能体 $i$ 的动作将如何影响其 Q 值，根据集中式评论器判断。
- 我们将其乘以智能体 $i$ 执行者输出相对于其参数 (parameter) $\theta_i$ 的梯度。
- 我们使用 Adam 等优化器应用此梯度来更新执行者参数 $\theta_i$。

这个更新规则有效地促使执行者 $\mu_i$ 输出那些被集中式评论器 $Q_i$ 认为更好的动作，同时考虑联合背景。请注意，计算此梯度仅需要知道 $\mathbf{x}$ 和所有智能体的当前策略 $\mu_j$；它不需要知道其他智能体的评论器参数。

### 使用集成进行策略推断（可选改进）

一个挑战是策略在训练过程中可能快速变化，这可能误导其他智能体。MADDPG 提出了一种改进：为每个智能体学习一个包含 $K$ 个策略的集成。在更新智能体的策略时，在目标 $y_i$ 和策略梯度 $J(\theta_i)$ 中考虑来自其他智能体的集成平均动作。这可以产生更稳定的策略，减少对其他智能体瞬态行为的利用。然而，这会增加复杂性。

### MADDPG 的优点和缺点

**优点：**

- **处理连续动作：** 可直接应用于具有连续动作空间的问题，继承了 DDPG 的这一优势。
- **普遍适用性：** 适用于协作、竞争或混合环境，无需对奖励或价值函数进行结构性假设。
- **解决非平稳性：** 集中式评论器在训练期间提供稳定的学习信号。
- **CTDE：** 使用全局信息进行更好的训练，同时允许高效的分散式执行。

**缺点：**

- **评论器的可扩展性：** 集中式评论器的输入包含所有智能体的观察和动作。这个输入维度会随智能体数量 $N$ 线性增长，可能使得当 $N$ 非常大时，评论器难以训练。
- **需要全局信息：** 训练需要访问所有智能体的观察和动作，以形成评论器的输入。这并非总是可行。
- **超参数 (parameter) (hyperparameter)敏感性：** 像 DDPG 一样，MADDPG 可能对超参数（学习率、网络大小、探索噪声等）敏感。

### 总结

MADDPG 为多智能体强化学习 (reinforcement learning)提供了一个强大且广泛使用的框架，在涉及连续动作和混合智能体交互的场景中尤其有效。通过在训练期间采用集中式评论器，它巧妙地避开了困扰简单独立学习方法的非平稳性问题。尽管它在集中式评论器输入大小方面存在可扩展性限制，但其清晰的原理和在各种基准测试上的优秀表现使其成为 MARL 工具集中的一个重要算法。理解 MADDPG 为处理智能体必须学习协调或竞争策略的复杂多智能体问题提供了坚实的依据。

## 参考资料

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Pages: 6379-6390; DOI: [10.48550/arXiv.1706.02275](https://doi.org/10.48550/arXiv.1706.02275)
  介绍MADDPG算法的原始论文，详细说明了其架构以及用于多智能体环境的集中式训练与分布式执行框架。
- [Continuous Control with Deep Reinforcement Learning](https://doi.org/10.48550/arXiv.1509.02971) — Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra (2016)
  Journal: International Conference on Learning Representations (ICLR); Pages: 1-14; DOI: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971)
  介绍了深度确定性策略梯度（DDPG）算法，该算法是MADDPG的基础单智能体演员-评论家方法。

---

[上一节](08-%E4%BB%B7%E5%80%BC%E5%88%86%E8%A7%A3%E6%96%B9%E6%B3%95%20%28VDN%2C%20QMIX%29.md) · [下一节](10-%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.md)
