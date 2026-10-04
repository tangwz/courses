---
course: "intermediate-reinforcement-learning"
chapter: "deep-q-networks-dqn"
lesson: "neural-networks-q-approximation"
sourceId: 5099
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/neural-networks-q-approximation"
title: "使用神经网络进行Q值近似"
description: "神经网络如何表示动作值函数Q(s, a)。"
order: 2
plots: []
sourceHash: "89958b8991f06f5c93cac03968d535c332731c511b5eeb0baddf3aa471728666"
sourceCorrections: []
---

当处理状态数量非常多的环境时，比如那些由图像或复杂特征向量 (vector)表示的环境，存储Q值的表格方法会失效。试想一下，要为屏幕上每一种可能的像素配置创建一个表格条目。由于内存限制以及访问每个状态以学习其值的几乎不可能，这根本不可行。此外，表格方法不能进行归纳。如果智能体遇到一个之前从未见过的状态，即使它与一个已知状态非常相似，也无法对其值进行估算。

为应对这些挑战，我们采用函数近似。我们不再为表格中的每个$Q(s, a)$对存储精确值，而是使用一个带有可学习参数 (parameter)的函数来*估算*这些值。我们的目的是找到一个由向量$\theta$参数化的函数$Q(s, a; \theta)$，它能近似真实的动作值函数$Q(s, a)$。

神经网络 (neural network)特别适合这项任务。它们是强大的函数近似器，能够学习输入和输出之间复杂、非线性的关系。更重要的是，它们擅长处理高维输入，例如游戏屏幕的原始像素数据或机器人的传感器读数，并且可以自动从这些数据中习得有用的特征。通过使用神经网络，我们的目标是学习一个参数向量$\theta$（表示网络的权重 (weight)和偏置 (bias)），使得：


$$
Q(s, a) \approx Q(s, a; \theta)
$$


### 用于Q值估计的网络结构

我们如何构建神经网络 (neural network)以表示$Q(s, a; \theta)$？一种常见且有用的方法，特别适用于具有离散动作空间的环境（例如向左、向右、向上或向下移动），是设计一个网络，该网络以状态$s$作为输入，并输出一个Q值向量 (vector)，其中包含该状态下每个可能动作$a$的Q值。

- **输入层：** 接收状态表示$s$。这可以是像素值的扁平化向量、一组经过设计的特征，或者在处理图像时，是卷积层的输出。
- **隐藏层：** 一个或多个层（例如，全连接层、卷积层），处理输入状态表示，提取相关特征并学习复杂的模式。这些层的复杂性和类型取决于状态空间的性质。
- **输出层：** 生成一个向量，其中每个元素对应于特定动作的估计Q值。如果环境中存在$N$个可能的动作，输出层将有$N$个神经元，其中第$i$个神经元输出估计值$Q(s, a_i; \theta)$。

> 该图展示了神经网络如何以状态$s$作为输入，并输出该状态下每个可能动作$a_i$的估计Q值。网络的参数 (parameter)由$\theta$表示。

这种架构很高效，因为它使得我们能通过一次网络前向传播来计算给定状态下所有动作的Q值。这对于动作选择很有帮助，在动作选择中，我们通常需要找到具有最高Q值的动作（即$\arg\max_a Q(s, a; \theta)$）。

### 泛化的能力

使用神经网络 (neural network)作为函数近似器最重要的优势是**泛化能力**。由于网络学习状态空间中的底层模式，即使对于训练期间未曾遇到的状态，只要它们与已见过的状态相似，网络也能生成合理的Q值估算。如果两个状态$s_1$和$s_2$由相似的输入向量 (vector)表示，网络很可能会为它们生成相似的Q值输出。这使得智能体能够比表格方法更有效地运用过往经验，从而在规模大的状态空间中实现更快的学习。

例如，在雅达利游戏中，网络可能会学习到某些视觉模式（例如接近的敌人）与负面结果相关联，而无论其在屏幕上的确切像素位置如何。它学习状态的一种压缩的、有用的表示，这种表示包含决策所需的重要信息。

通过用神经网络替代Q表，我们为深度Q网络（DQN）打下了基础。后续章节将详细说明我们如何使用Q学习中调整后的技术来训练该网络的参数 (parameter)$\theta$，并结合经验回放和目标网络等方法，以确保学习的稳固和高效。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  提供了强化学习中函数逼近的全面理论基础，详细阐述了从表格方法到参数化函数的演变，并介绍了神经网络的应用。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇开创性论文介绍了深度Q网络（DQN），展示了深度神经网络如何成功地从高维感官输入中直接逼近Q值，并在Atari游戏中实现了人类水平的控制。
