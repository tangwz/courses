---
course: "advanced-reinforcement-learning"
chapter: "deep-q-networks"
lesson: "experience-replay"
sourceId: 3367
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/experience-replay"
title: "经验回放机制"
description: "了解经验回放如何打破相关性并提高DQN中的数据效率。"
order: 3
plots: []
sourceHash: "60a46cc6fa4f83bcf7a686639f6835d68c3a2a8cbef10cc5d97dbd26b5b77de5"
sourceCorrections: []
---

有效训练深度神经网络 (neural network)通常依赖于输入数据点是独立同分布（i.i.d.）的假设。然而，当强化学习 (reinforcement learning)智能体与环境顺序交互时，生成的经验 $(s_t, a_t, r_t, s_{t+1})$ 绝非i.i.d.。连续状态高度相关，并且经验分布随智能体策略演变而变化。直接在这些顺序的、相关的经验上训练深度Q网络，常导致不稳定和收敛不良。网络很容易过拟合 (overfitting)于最近的、相关的轨迹，遗忘过去的、可能有价值的经验。

经验回放是一种受生物学启发的机制，与DQN一同引入以应对这些难题。智能体并非仅使用最近的转换进行更新，而是将其经验存储在一个大型数据结构中，该结构称为回放缓冲区（或回放记忆），通常表示为 $D$。该缓冲区通常具有固定容量，并像FIFO（先进先出）队列一样运行：当缓冲区满并有新经验到来时，最旧的经验会被移除。

核心过程工作方式如下：

1. **存储：** 在每个时间步 $t$，在智能体与环境交互并观测到转换 $(s_t, a_t, r_t, s_{t+1})$ 后，此转换元组会存储到回放缓冲区 $D$ 中。
2. **抽取：** 我们不基于当前转换 $(s_t, a_t, r_t, s_{t+1})$ 更新Q网络，而是从整个缓冲区 $D$ 中均匀随机抽取一小批转换 $(s_j, a_j, r_j, s_{j+1})$。
3. **更新：** Q网络参数 (parameter) $\theta$ 使用梯度下降 (gradient descent)，基于此小批次计算的损失进行更新。

> 经验回放过程：转换存储在缓冲区中，并从此缓冲区随机抽取小批次以进行Q网络更新。

这种随机抽取具有两个主要益处：

1. **打破相关性：** 通过从大量历史转换中随机抽取，更新基于多样化的经验集合。这打破了存在于顺序交互中的强大时间相关性，使得训练数据更接近稳定随机梯度下降更新所需的i.i.d.假设。这大大稳定了学习过程。
2. **提高数据效率：** 每个经验转换都可以被多次抽取并用于网络更新。这使得智能体能从每次交互中学习更多，这在数据收集昂贵或耗时的场景中尤其有利。

对于小批次 $B$ 中的每个抽取转换 $(s_j, a_j, r_j, s_{j+1})$，计算目标值 $y_j$，通常采用一个单独的目标网络（接下来会讨论）其参数为 $\theta^-$，以进一步稳定训练：


$$
y_j = r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta^-)
$$


如果 $s_{j+1}$ 是终止状态，则 $y_j = r_j$。损失，通常是均方误差（MSE），随后在小批次上计算：


$$
L(\theta) = \frac{1}{|B|} \sum_{j \in B} \left( y_j - Q(s_j, a_j; \theta) \right)^2
$$


主Q网络的参数 $\theta$ 随后使用梯度下降，基于此损失 $L(\theta)$ 进行更新。

回放缓冲区的大小是一个重要的超参数 (hyperparameter)。更大的缓冲区存储更具多样性的经验集合，减少相关性，但可能包含来自旧策略的过时信息。更小的缓冲区能更快适应最近的策略变化，但可能遭受更强的相关性影响。典型的缓冲区大小范围从数万到数百万次转换，取决于任务的复杂性和内存限制。

经验回放是促成DQN成功的重要组成部分，它将带有非线性函数逼近器的Q学习从一种通常不稳定的技术，转变为一种强大且广泛适用的深度强化学习算法。虽然简单的均匀抽取是标准做法，但后续的改进，例如优先级经验回放（PER）（我们将在本章稍后讨论它），进一步优化了抽取策略以提高效率。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  引入了深度Q网络（DQN）算法，开创性地将经验回放用于强化学习环境中深度神经网络的稳定高效训练。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本标准教材，全面涵盖强化学习，包括Q学习、深度Q网络和经验回放如何在函数逼近器中稳定学习的详细解释。
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver (2016)
  Journal: ICLR; DOI: [10.48550/arXiv.1511.05952](https://doi.org/10.48550/arXiv.1511.05952)
  该论文在引入优先经验回放的同时，讨论了均匀经验回放的特点和局限性，强调了采样策略如何进一步提高学习效率。
