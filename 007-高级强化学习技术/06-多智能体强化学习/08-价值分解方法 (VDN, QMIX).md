# 价值分解方法 (VDN, QMIX)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/value-decomposition-marl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

最大化共享的团队奖励是协作式多智能体环境中的一个共同目标。集中训练去中心化执行 (CTDE) 方法提供了一种实用办法：智能体可以潜在地获取全局信息共同学习，但在执行时必须仅根据各自的局部观测独立行动。CTDE 中的一个主要难题是如何有效地协调行动。如果学习一个集中式的联合动作价值函数 $Q_{tot}(\mathbf{s}, \mathbf{a})$（其中 $\mathbf{s}$ 是全局状态或训练期间可用的某种表示，$\mathbf{a}$ 是联合动作 $(a_1, ..., a_N)$），那么如何从中提取智能体可以根据其局部观测 $o_i$ 使用的去中心化策略 $\pi_i(a_i | o_i)$，就是一个主要问题。

对 $Q_{tot}$ 进行关于 $\mathbf{a}$ 的全局 `argmax` 运算通常在计算上不可行，尤其当智能体和动作的数量增加时。价值分解方法通过学习单个智能体的价值函数 $Q_i(o_i, a_i)$，然后以结构化的方式组合它们来近似 $Q_{tot}$，从而确保最大化 $Q_{tot}$ 可以通过每个智能体最大化自己的 $Q_i$ 来实现。

### 价值分解网络 (VDN)

价值分解最简单的方法是价值分解网络 (VDN)。VDN 假定联合动作价值函数可以加性分解为单个智能体的价值函数：


$$
Q_{tot}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^{N} Q_i(o_i, a_i)
$$


这里，$N$ 是智能体的数量。每个 $Q_i$ 通常由一个神经网络 (neural network)表示，该网络将智能体 $i$ 的局部观测 $o_i$ 和动作 $a_i$ 作为输入（通常，在给定 $o_i$ 的情况下，该网络会输出所有可能动作 $a_i$ 的 Q 值）。全局 $Q_{tot}$ 只是这些单个 Q 值的总和。

**它如何支持 CTDE：**

1. **集中训练：** 在训练期间，我们可以通过对单个 $Q_i$ 值求和来计算 $Q_{tot}$。损失（例如，Q-learning 的 TD 误差）是根据 $Q_{tot}$ 和全局奖励计算的，梯度通过求和操作反向传播 (backpropagation)，以更新每个单个 $Q_i$ 网络。这通常需要获取全局状态 $\mathbf{s}$，或者至少是联合动作 $\mathbf{a}$ 和全局奖励 $r$。
2. **去中心化执行：** 在执行时，每个智能体 $i$ 只需其局部观测 $o_i$。它会计算其 $Q_i(o_i, \cdot)$ 对于所有可能动作的值，并贪婪地选择动作：
   
   $$
   a_i^* = \arg\max_{a_i} Q_i(o_i, a_i)
   $$
   
   由于全局 $Q_{tot}$ 仅仅是一个总和，单独最大化每个 $Q_i$ 可以确保联合动作 $\mathbf{a}^* = (a_1^*, ..., a_N^*)$ 能够最大化 $Q_{tot}$。

> VDN 架构。每个智能体都有一个独立的 Q 网络。在训练期间，它们的输出被求和以形成 $Q_{tot}$，用于损失计算。在执行期间，每个智能体根据自己的 $Q_i$ 贪婪地行动。

**局限：**
VDN 的主要局限是其严格的加性假定。它只能表示每个智能体的贡献独立于其他智能体动作的联合动作价值函数。这使得它无法对更复杂的协作场景进行建模，在这些场景中，一个智能体动作的价值严重依赖于其他智能体正在做什么。

### QMIX：单调价值函数分解

QMIX (Q-Mixing) 解决了 VDN 在表示上的局限，同时保持了方便的去中心化执行特性。QMIX 没有采用简单的求和，而是使用一个*混合网络*来组合单个的 $Q_i$ 值以形成 $Q_{tot}$。


$$
Q_{tot}(\mathbf{s}, \mathbf{a}) = f(\{Q_i(o_i, a_i)\}_{i=1}^N, \mathbf{s})
$$


函数 $f$，由混合网络表示，被设计为满足一个重要的*单调性约束*：


$$
\frac{\partial Q_{tot}}{\partial Q_i} \ge 0 \quad \forall i
$$


该约束足以确保如果一个智能体增加其单个 $Q_i$ 值，全局 $Q_{tot}$ 值将增加或保持不变，但绝不会减少。

**单调性为何重要：**
单调性约束足以确保对 $Q_{tot}$ 进行全局 `argmax` 运算会得到与对每个 $Q_i$ 分别进行 `argmax` 运算相同的结果：


$$
\arg\max_{\mathbf{a}} Q_{tot}(\mathbf{s}, \mathbf{a}) = (\arg\max_{a_1} Q_1(o_1, a_1), ..., \arg\max_{a_N} Q_N(o_N, a_N))
$$


这表示 QMIX 保留了 VDN 中去中心化执行的便利性，即每个智能体只选择最大化其自身学习到的 $Q_i$ 的动作。

**混合网络架构：**
混合网络通过具有非负权重 (weight)来强制执行单调性约束。它通常将单个 $Q_i$ 网络的输出作为输入。最重要的是，混合网络本身的权重和偏置 (bias)是由单独的*超网络*生成的，这些超网络接收全局状态 $\mathbf{s}$ 作为输入。这使得单个 $Q_i$ 值组合的方式能够依赖于状态提供的整体上下文 (context)，从而使 QMIX 比 VDN 更具表示能力。权重的非负性通常通过在生成权重的超网络输出上使用绝对激活函数 (activation function)或 ReLU 来强制实现。

> QMIX 架构。单个智能体 Q 网络（$Q_1$ 到 $Q_N$）去中心化运行。它们的输出输入到一个中央混合网络。混合网络的参数 (parameter)（权重和偏置）由基于全局状态 $s$ 的超网络生成，从而保证单调性（$\partial Q_{tot}/\partial Q_i \ge 0$），并允许复杂、依赖于状态的组合，同时保持可行的去中心化执行。

**相对于 VDN 的优势：**
QMIX 比 VDN 能够表示更丰富的协作式多智能体强化学习 (reinforcement learning)问题，因为混合网络可以学习单个智能体价值的复杂非线性组合，这些组合是基于全局状态的。唯一的结构约束是单调性，这比纯粹的加性约束要宽松得多。

**总结：**
VDN 和 QMIX 是协作式多智能体强化学习的 CTDE 框架内重要的基于价值的方法。它们学习单个智能体 Q 函数 $Q_i$，这些函数支持去中心化执行。

- **VDN** 使用简单的求和，这使得它易于实现，但限制了其表示能力。
- **QMIX** 采用了一个带有单调性约束的状态依赖混合网络，提供了更强的表示能力，同时保留了重要特性：最大化单个 $Q_i$ 函数能够最大化联合 $Q_{tot}$。

这些方法通过分解团队的价值函数，提供了协调智能体团队的有效方式，为协作式多智能体强化学习中的许多高级技术提供了重要支持。

## 参考资料

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson (2018)
  Journal: International Conference of Machine Learning 2018; Volume: 80; Pages: 4295-4304; DOI: [10.48550/arXiv.1803.11485](https://doi.org/10.48550/arXiv.1803.11485)
  这篇论文介绍了QMIX，一种利用混合网络和单调性约束的价值分解方法，相比VDN能实现更复杂的依赖状态的价值分解。
- [A Survey of Multi-Agent Reinforcement Learning from a Cooperative Perspective](https://arxiv.org/abs/2101.09505) — Kai Zhang, Longbo Huang, Jiaxun Cui, Boyuan Chen (2021)
  Journal: arXiv preprint arXiv:2101.09505; DOI: [10.48550/arXiv.2101.09505](https://doi.org/10.48550/arXiv.2101.09505)
  这份综述全面概述了合作多智能体强化学习，详细介绍了包括VDN和QMIX在内的多种价值分解方法。

---

[上一节](07-%E9%9B%86%E4%B8%AD%E5%BC%8F%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%8E%BB%E4%B8%AD%E5%BF%83%E5%8C%96%E6%89%A7%E8%A1%8C%20%28CTDE%29.md) · [下一节](09-%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E6%B7%B1%E5%BA%A6%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%20%28MADDPG%29.md)
