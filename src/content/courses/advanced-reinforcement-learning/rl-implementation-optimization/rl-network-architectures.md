---
course: "advanced-reinforcement-learning"
chapter: "rl-implementation-optimization"
lesson: "rl-network-architectures"
sourceId: 3495
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-8-rl-implementation-optimization/rl-network-architectures"
title: "强化学习中的神经网络架构"
description: "用于策略和值函数设计的网络架构（CNN、RNN、Transformer）的最佳实践。"
order: 1
plots: []
sourceHash: "13505d5aa0ff4cd0006d254f1acdd4a93cfc8744f0ac0ba34c57e16ae26a415a"
sourceCorrections: []
---

在实现深度强化学习 (reinforcement learning)智能体时，选择合适的神经网络 (neural network)架构非常重要。就像建筑物的地基决定了它的稳定性和高度一样，网络结构极大地影响着智能体的学习能力、训练过程中的稳定性以及整体表现。核心强化学习算法决定了智能体*如何*学习，而网络架构则决定了它*能表示什么*以及*如何高效地*处理来自环境的信息。

本节主要讨论设计这些网络时的实际考量，会根据强化学习任务的具体要求进行调整，并建立在您已有的深度学习 (deep learning)组件知识之上。

### 处理环境观测

设计网络的第一步是处理输入：环境提供的观测值（$o_t$）。观测空间的性质在很大程度上决定了网络的初始层。

- **向量 (vector)观测：** 对于状态表示为固定大小数值特征向量（例如，倒立摆位置和速度、机器人关节角度）的环境，标准多层感知机（MLP）通常很有效。这些通常由若干个带有ReLU等非线性激活函数 (activation function)的全连接（密集）层组成。对于MLP输入，对输入特征进行缩放（归一化 (normalization)或标准化）对于稳定训练通常非常重要。
- **图像观测：** 当智能体通过图像（例如，Atari游戏屏幕的像素，或机器人中的摄像头反馈）感知环境时，卷积神经网络 (neural network)（CNN）是标准选择。受计算机视觉成功经验的启发，常见的模式是堆叠多个卷积层（通常带有ReLU激活和可能的池化层）以提取特征的空间层次结构，之后是一个或多个全连接层，以产生最终输出（Q值或策略参数 (parameter)）。

  - **预处理：** 原始像素很少直接使用。常见的预处理步骤包括将图像转换为灰度、下采样，以及堆叠连续帧（例如4帧），以向网络提供关于运动和动态的信息。
  - **示例结构：** `Input (stacked frames) -> Conv2D -> ReLU -> Conv2D -> ReLU -> Conv2D -> ReLU -> Flatten -> Dense -> ReLU -> Output Layer`。
- **序列观测：** 如果观测具有固有时间依赖性，且无法通过简单的帧堆叠完全捕捉，或者输入本身是序列化的（例如，基于文本游戏中的文本、时间序列数据），则可以使用循环神经网络（RNN），如长短期记忆（LSTM）或门控循环单元（GRU）。这些网络维护内部隐藏状态，允许它们随时间整合信息。使用RNN会增加训练的复杂性，需要时间反向传播 (backpropagation)（BPTT）等技术，并谨慎处理隐藏状态，尤其是在使用经验回放时。
- **Transformer架构：** 对于涉及非常长距离依赖关系或结构化数据（如自然语言指令或复杂关系环境）的任务，使用自注意力 (self-attention)机制 (attention mechanism)的Transformer网络变得越来越可行。它们擅长建模序列中远距离点之间的依赖关系，但与CNN或RNN相比，计算成本和数据需求更高。

### 生成智能体动作

网络的输出层必须与环境的动作空间（$a_t$）相匹配。

- **离散动作空间：** 对于具有有限动作集（例如，“左”、“右”、“开火”）的环境，网络通常输出一个向量 (vector)，其中每个元素对应一个动作。

  - **基于值的方法（例如，DQN）：** 输出层为状态$s$中的每个可能动作$a$提供估计的Q值 $Q(s, a)$。线性输出层很常见。
  - **基于策略的方法（例如，A2C、PPO）：** 输出层为每个动作提供logits（未归一化 (normalization)的对数概率）。这些logits通常通过Softmax函数形成一个分类概率分布 $\pi(a|s)$，从中抽取动作。
- **连续动作空间：** 对于动作是连续向量（例如，电机扭矩、转向角度）的环境，网络输出定义动作概率分布的参数 (parameter)。

  - **确定性策略（例如，DDPG）：** 网络直接输出动作向量 $a = \mu(s)$。通常将`tanh`等激活函数 (activation function)应用于输出层，以将动作限制在特定范围（例如[-1, 1]）内，然后可以根据环境要求进行缩放。
  - **随机策略（例如，PPO、SAC）：** 网络通常输出某个分布的参数，最常见的是高斯分布的均值 $\mu(s)$ 和标准差 $\sigma(s)$（或为稳定性考虑使用对数标准差）。动作然后从 $\mathcal{N}(\mu(s), \sigma(s)^2)$ 中采样。必须注意确保标准差保持正值（例如，通过使用`softplus`激活函数或对网络输出进行指数运算）。在这些Actor-Critic设置中，评论家网络仍然输出一个标量值估计（$V(s)$或$Q(s, a)$）。

### Actor-Critic 方法的常见网络结构

在Actor-Critic算法中，你需要分别用于策略（actor）和值函数（critic）的网络。一个常见的设计选择是它们之间是否共享参数 (parameter)。

- **独立网络：** 为actor和critic使用完全独立的网络。这更易于实现和调试，并避免了策略和值函数学习目标之间潜在的负面干扰。然而，它可能样本效率较低，因为与两个任务相关的特征是单独学习的。
- **共享主干，分离头部：** 使用共同的初始层（例如，用于图像处理的CNN层）从观测中提取共享特征表示。然后，为actor（策略输出）和critic（值输出）附加独立的输出层（“头部”）。这鼓励学习共享表示，可能提高样本效率和泛化能力。然而，平衡共享部分和独立部分的梯度和学习率可能具有挑战性。

> Actor-Critic方法中独立网络与共享主干架构的比较。

### Dueling架构（基于值）

如第二章所述，Dueling网络架构是一种专门的修改，主要用于DQN等基于值的方法。它在特征提取层之后将网络分成两个流：

1. **值流：** 估计状态值函数 $V(s)$。输出一个标量。
2. **优势流：** 估计每个动作的优势函数 $A(s, a)$。输出一个维度等于动作数量的向量 (vector)。

这些流随后组合起来以生成最终的Q值。一个常见的聚合方法是：


$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
$$


这种中心化操作确保了可识别性并提高了稳定性。直观理解是，将状态的内在价值与动作特异性优势分开，可以带来更有效的学习，尤其是在某个状态的价值不强烈依赖于所采取的具体动作时。

> Dueling网络架构的结构，它在组合为Q值之前分离了值和优势的估计。

### 架构设计原则与技巧

虽然没有适用于所有强化学习 (reinforcement learning)问题的“最佳”架构，但以下一些指导原则有所帮助：

- **从简单开始：** 从已知的、相对较浅的、适用于类似问题的架构（例如，DQN的标准Atari CNN）开始。仅当性能停滞不前，并且你怀疑表示能力是瓶颈时，才增加复杂性（深度、宽度、循环性）。
- **输入/输出匹配：** 确保输入层与观测空间匹配，输出层与动作空间和算法要求（Q值、策略分布参数 (parameter)）匹配。
- **归一化 (normalization)：** 输入观测的归一化通常很重要。可以应用诸如滑动平均/标准差滤波器或批归一化等技术，但在强化学习中，由于策略演变和经验回放缓冲区可能导致数据分布非平稳，使用层归一化或谨慎使用批归一化。奖励缩放或裁剪也是常见做法。
- **激活函数 (activation function)：** ReLU因其简单性和在缓解梯度消失方面的有效性，仍然是隐藏层的常见选择。`tanh`常用于限制连续动作的范围。选择适合所需特定输出的激活函数（例如，Q值的线性函数，标准差的`softplus`函数）。
- **初始化：** 标准的权重 (weight)初始化技术（例如，正交初始化、Xavier、He）是重要的起点。有时，将最终策略层初始化为非常小的权重有助于鼓励初始的探索。
- **参考文献：** 参考现有研究。论文通常会详细说明用于基准任务的架构。从一个已被验证的架构开始可以显著增加成功实现的几率。

选择和优化网络架构是一个迭代过程，通常需要结合超参数 (hyperparameter)调整（下一节介绍）进行实验，以实现给定强化学习任务的最佳性能。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了基础的深度Q网络（DQN）架构，将卷积神经网络应用于图像观测以实现离散动作控制。
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) — Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Volume: 48; Pages: 1995-2003; DOI: [10.48550/arXiv.1511.06581](https://doi.org/10.48550/arXiv.1511.06581)
  介绍了决斗网络架构，该架构分离了状态值和动作优势的估计，用于基于价值的深度强化学习。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  提出了近端策略优化（PPO）算法，并阐明了适用于Actor-Critic方法的网络架构，包括连续动作空间和共享特征层的设计。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  对深度学习进行了全面概述，涵盖了多层感知器、卷积神经网络、循环神经网络等基础架构、激活函数及正则化方法，对深度强化学习有用。
