---
course: "intermediate-reinforcement-learning"
chapter: "deep-q-networks-dqn"
lesson: "dqn-algorithm-architecture"
sourceId: 5101
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/dqn-algorithm-architecture"
title: "DQN 算法架构"
description: "深度 Q 网络核心算法结构的详细解析。"
order: 3
plots: []
sourceHash: "e87a90766f7430bcd31fa4c0cd06146e3939e5329d7bd97837a6ace71791c199"
sourceCorrections: []
---

表格 Q 学习在处理具有海量状态的环境时会遇到瓶颈。想象一下，尝试为屏幕上像素的每种可能配置存储一个 Q 值，这根本不可行。深度 Q 网络 (DQN) 通过用深度神经网络 (neural network)（通常称为 Q 网络）取代 Q 表来解决这个问题。该网络充当函数逼近器，学习估计 Q 值。

### Q 网络结构

我们不再在表格中查找值，而是将智能体的当前状态表示 $s$ 输入到神经网络 (neural network)中。该网络的作用是输出在该状态 $s$ 中*所有可能的动作* $a$ 的动作-值函数 $Q(s, a)$ 的估计值。

- **输入：** 网络的输入层接收状态表示。这可以是游戏屏幕的原始像素数据、机器人的传感器读数，或描述环境当前状况的其他特征。可能需要进行预处理，以便为网络适当地规范化或组织此输入。
- **隐藏层：** 这些是标准神经网络层（例如，用于图像输入的卷积层、全连接层），它们处理输入状态并学习与估计动作值相关的复杂特征和关系。
- **输出：** 输出层生成一个值向量 (vector)。此向量中的每个元素对应于针对一个特定动作的估计 Q 值。例如，如果一个环境有 4 个可能的动作（上、下、左、右），网络将输出 4 个值：$Q(s, \text{上})$、$Q(s, \text{下})$、$Q(s, \text{左})$ 和 $Q(s, \text{右})$。

令 $\theta$ 表示该神经网络的参数 (parameter)（权重 (weight)和偏置 (bias)）。我们可以将网络在状态 $s$ 和动作 $a$ 上的输出表示为 $Q(s, a; \theta)$。

> 该图展示了 DQN 智能体在决策时的基本流程。当前状态 $s_t$ 被输入到 Q 网络中，Q 网络输出所有动作的估计 Q 值。然后，动作选择策略（如 epsilon-贪婪策略）利用这些 Q 值来选择在环境中执行的动作 $a_t$。

### 学习过程：Q 学习的调整

我们如何训练这个 Q 网络？我们调整 Q 学习更新规则中的核心思想。回想贝尔曼方程，它表达了当前状态-动作对的 Q 值与下一状态的 Q 值之间的关系：


$$
Q^*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t=s, A_t=a]
$$


在标准 Q 学习中，我们迭代地更新 Q 表项以逼近这个目标值。对于 DQN，我们将其视为一个监督学习 (supervised learning)问题。我们希望网络的预测 $Q(S_t, A_t; \theta)$ 能够逼近一个源自贝尔曼方程的目标值。

对于给定的转换 $(S_t, A_t, R_{t+1}, S_{t+1})$，目标值 $y_t$ 是根据获得的奖励和*下一*状态 $S_{t+1}$ 的最大估计 Q 值计算的：


$$
y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)
$$


这里有一个重要点需要注意：在这个基本表述中，我们使用*同一个* Q 网络（带有参数 (parameter) $\theta$）来预测当前状态-动作对的 Q 值 ($Q(S_t, A_t; \theta)$)，并估计下一状态的最大 Q 值 ($\max_{a'} Q(S_{t+1}, a'; \theta)$)，这都是计算目标 $y_t$ 所需的。正如我们之后会看到的，这种耦合可能导致训练过程中的不稳定。

### 损失函数 (loss function)

网络通过最小化其预测 $Q(S_t, A_t; \theta)$ 和目标值 $y_t$ 之间的差异来学习。损失函数的一个常见选择是均方误差（MSE），在一批转换上取平均：


$$
L(\theta) = \mathbb{E} \left[ (y_t - Q(S_t, A_t; \theta))^2 \right]
$$


代入 $y_t$ 的定义后：


$$
L(\theta) = \mathbb{E} \left[ \left( R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta) - Q(S_t, A_t; \theta) \right)^2 \right]
$$


然后我们使用标准的梯度下降 (gradient descent)算法（如 Adam 或 RMSprop）计算此损失函数相对于网络参数 (parameter) $\theta$ 的梯度 $\nabla_{\theta} L(\theta)$，并更新权重 (weight)以最小化误差：


$$
\theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta)
$$


其中 $\alpha$ 是学习率。

这种架构使得智能体即使在高维状态空间中也能学习 Q 值，这凭借了深度神经网络 (neural network)的表示能力。然而，直接训练这种设置常常遇到重大的挑战，这些挑战与相关联的样本和移动的目标值有关。接下来的部分将介绍两种基本技术：经验回放和目标网络，它们专门设计用于稳定深度 Q 网络的训练过程。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍深度Q网络（DQN）算法的原始论文，详细说明了其架构和学习过程，使其在雅达利游戏中达到人类水平的表现。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本内容全面的教科书，提供了Q学习、使用神经网络进行函数逼近的理论背景，并包含一章专门讨论深度Q网络（DQN）。
