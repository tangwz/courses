---
course: "intro-to-reinforcement-learning"
chapter: "introduction-deep-q-networks-dqn"
lesson: "combining-q-learning-deep-learning"
sourceId: 1658
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/combining-q-learning-deep-learning"
title: "Q学习与深度学习的结合"
description: "说明深度神经网络作为Q学习函数逼近器的应用缘由。"
order: 1
plots: []
sourceHash: "e7fb03d634abaec20f185a0e0577fa0bf74d7389ba227c9b2b5912ed09dab3e1"
sourceCorrections: []
---

“在上一章中，我们了解了函数逼近如何帮助强化学习 (reinforcement learning)代理在不同状态间泛化知识，从而克服了表格方法在大型状态空间中的局限性。虽然线性函数逼近带来了一些改进，但许多问题涉及状态与价值之间高度复杂的关系，线性模型难以捕捉这些关系。设想这样的任务，例如直接从屏幕像素玩视频游戏，或者根据高维传感器输入控制机器人。这里的状态空间极其庞大，最优策略可能取决于输入中复杂的非线性模式。”

为了处理大规模且包含复杂非线性模式的问题，深度神经网络 (neural network)展现出其作用。神经网络擅长从高维输入中学习复杂的层次化特征和非线性映射。通过使用神经网络来逼近动作价值函数 $Q(s, a)$，我们能够在视觉信息丰富或状态表示高度复杂的环境中，学习到有效的策略。

我们称这种方法为深度Q学习（Deep Q-Learning），所用的神经网络通常被称为深度Q网络（DQN）。我们不将Q值存储在表格中或使用简单的线性函数，而是使用一个神经网络，其参数 (parameter)（权重 (weight)和偏置 (bias)）统称为 $\theta$。这个网络将状态表示 $s$ 作为输入，并输出该状态下每个可能动作 $a$ 的Q值估计。一种常见的架构是输出一个Q值向量 (vector)，每个动作一个条目：

$Q(s, \cdot; \theta) \approx Q^*(s, \cdot)$

此处，$Q(s, \cdot; \theta)$ 代表网络在给定参数 $\theta$ 下对状态 $s$ 的输出，旨在逼近真实的最佳动作价值函数 $Q^*(s, \cdot)$。例如，如果一个代理正在处理游戏屏幕的图像（状态 $s$），DQN可能会输出按下左键、按下右键、跳跃或射击的预期累积未来奖励。

因此，目标是训练网络，这意味着我们需要找到参数 $\theta$，使网络的输出 $Q(s, a; \theta)$ 能够良好逼近最优 $Q^*(s, a)$。我们如何做到这一点呢？我们可以采用Q学习的原理和贝尔曼最优方程。

回顾一下，标准的Q学习更新依赖于TD目标：$y = R + \gamma \max_{a'} Q(S', a')$。这个目标代表了基于获得的奖励 ($R$) 和从下一状态 ($S'$) 估计的最大未来价值的最优Q值估计。在深度Q学习中，我们将网络的训练视为一个监督学习 (supervised learning)问题。对于给定的转换 $(S, A, R, S')$，网络预测当前的Q值 $Q(S, A; \theta)$。我们使用奖励 $R$ 和网络自身的对下一状态 $S'$ 最大Q值的估计来计算目标值 $y$。

理想情况下，我们希望网络的预测 $Q(S, A; \theta)$ 与这个目标 $y$ 匹配。我们可以定义一个损失函数 (loss function)，用于衡量预测与目标之间的差异。一个常见的选择是均方误差（MSE）：


$$
L(\theta) = \left( y - Q(S, A; \theta) \right)^2
$$


$$
L(\theta) = \left( \underbrace{R + \gamma \max_{a'} Q(S', a'; \theta)}_{\text{目标Q值 (y)}} - \underbrace{Q(S, A; \theta)}_{\text{预测Q值}} \right)^2
$$


我们的目标是最小化关于网络参数 $\theta$ 的这个损失函数。我们可以通过使用梯度下降 (gradient descent)算法来实现这一点，例如随机梯度下降（SGD）或其变体如Adam。通过反复采样转换 $(S, A, R, S')$ 并调整网络权重 $\theta$ 以减少损失，网络逐渐学会逼近最优动作价值函数。

这种结合前景看好。它使我们能够将Q学习应用于那些由于状态空间庞大的规模和复杂性而以前无法处理的问题。然而，在标准RL循环中直接应用神经网络带来了一系列特有的挑战。代理与环境交互生成的数据具有序列化、相关联的特性，这违反了监督学习中常做的独立性假设。此外，目标值 $y$ 本身依赖于正在更新的网络参数 $\theta$，这可能导致训练期间的不稳定性。接下来的章节将讨论这些挑战，并介绍为DQN开发的核心技术：经验回放和目标网络，这些技术旨在稳定学习过程。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了深度Q网络（DQN），将Q学习与深度神经网络结合以实现成功的控制任务，确立了深度强化学习中的一项重要技术。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本标准教科书，解释了强化学习的原理，包括Q学习、贝尔曼方程和函数近似方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础书籍，提供深度神经网络、网络架构以及使用基于梯度的优化进行训练的背景知识。
