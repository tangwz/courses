# DQN训练的损失函数

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/dqn-loss-function)

[返回章节目录](README.md) · [返回课程目录](../README.md)

深度Q网络（DQN）的训练需要明确定义如何调整其主Q网络的参数 (parameter)。这一训练过程基于DQN架构，该架构包含经验回放和目标网络等机制。核心任务是调整由$\theta$表示的网络参数，使输出$Q(s, a; \theta)$能够逼近真实的最优动作价值函数$Q^*(s, a)$。

回想一下，在标准Q学习中，我们基于贝尔曼方程迭代更新$Q(s, a)$的估计值。我们将当前的估计值朝向一个目标值移动，该目标值源自观察到的奖励和下一个状态的估计值。DQN借鉴了这一基本思想，并将其调整用于使用神经网络 (neural network)进行函数逼近。

我们需要一种方法来量化 (quantization)网络对给定状态-动作对$(s, a)$的当前预测与目标值之间的误差。这个误差衡量由**损失函数 (loss function)**定义。训练的目标是最小化这个损失。

对于从经验回放缓冲区采样得到的转移$(s, a, r, s')$，我们的Q网络预测值$Q(s, a; \theta)$。目标值应该是什么？就像在Q学习中一样，目标旨在包含即时奖励$r$以及从下一个状态$s'$可能采取的最佳动作的折扣价值。然而，为了提升稳定性（如前一节关于目标网络所讨论的），我们使用带有参数$\theta^-$的*目标网络*来估计下一个状态的价值。

目标值，通常表示为$y$，计算方式如下：


$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$


这里，$\gamma$是折扣因子，而$\max_{a'} Q(s', a'; \theta^-)$表示*目标网络*对下一个状态$s'$在所有可能的后续动作$a'$上预测的最大Q值。如果$s'$是终止状态，则目标简单地是$y = r$。

有了预测值$Q(s, a; \theta)$和目标值$y$，我们现在可以定义损失。由于我们希望预测值尽可能接近目标值，一个自然的选择是\*\*均方误差（MSE）\*\*损失，这在回归问题中很常见。对于单个转移，平方误差是$(y - Q(s, a; \theta))^2$。

实际中，我们在从经验回放缓冲区$D$中采样得到的小批量转移上计算损失。我们旨在最小化的损失函数$L(\theta)$是这个平方误差在采样转移上的期望：


$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ (y - Q(s, a; \theta))^2 \right]
$$


或者，更实际地，对于$N$个转移的迷你批量$\{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$：


$$
L(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( (r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-)) - Q(s_i, a_i; \theta) \right)^2
$$


这个损失函数衡量的是目标值（使用目标网络计算）与主网络预测的Q值之间的平均平方差。

### 训练网络

训练过程涉及迭代执行以下步骤：

1. 从经验回放缓冲区$D$中采样一小批转移$(s, a, r, s')$。
2. 对于小批量中的每个转移：
   a. 使用目标网络计算目标值$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$。处理终止状态，此时$y=r$。
   b. 使用主网络计算预测值$Q(s, a; \theta)$。
3. 使用上述公式计算小批量中目标值与预测值之间的MSE损失。
4. 对损失$L(\theta)$相对于主网络参数 (parameter)$\theta$执行一步梯度下降 (gradient descent)。这会更新主Q网络的权重 (weight)以减少误差。


$$
\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
$$


其中$\alpha$是学习率。

需要注意的是，在计算梯度$\nabla_\theta L(\theta)$时，目标值$y$被视为固定常量。梯度只流经主网络参数$\theta$，而不流经目标网络参数$\theta^-$。这种通过使用独立的目标网络和经验回放实现的解耦，是稳定DQN训练过程的十分关键之处。如果没有这些方法，不断变化的目标和相关数据样本将使收敛变得困难。

定期地，在一定数量的训练步骤后，目标网络$\theta^-$的权重会更新以匹配主网络$\theta$的权重。这保证了目标值能够逐渐适应主网络所代表的改进策略，同时在较短的时间尺度内保持稳定性。

明白这个损失函数 (loss function)和训练过程对于实现和调整DQN智能体十分重要。在下一节中，我们将介绍一个实际实现。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇基础论文介绍了深度Q网络（DQN）架构，包括经验回放和目标网络的应用，并定义了使用Q学习损失函数的训练目标。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  这本广泛使用的教科书涵盖了强化学习的基本概念，包括Q学习、贝尔曼方程，并清晰地解释了这些原理如何应用于深度Q网络。
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — Josh Achiam (2018)
  这个在线资源提供了深度强化学习的实践介绍，详细解释了DQN算法、其损失函数和训练方法，并提供示例。

---

[上一节](05-%E5%9B%BA%E5%AE%9AQ%E7%9B%AE%E6%A0%87%20%28%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%29.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8CartPole%E4%B8%8A%E5%AE%9E%E7%8E%B0DQN.md)
