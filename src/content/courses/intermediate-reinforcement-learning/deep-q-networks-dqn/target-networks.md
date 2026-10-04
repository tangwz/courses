---
course: "intermediate-reinforcement-learning"
chapter: "deep-q-networks-dqn"
lesson: "target-networks"
sourceId: 5108
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/target-networks"
title: "固定Q目标 (目标网络)"
description: "使用单独的目标网络来稳定DQN训练。"
order: 5
plots: []
sourceHash: "c205075cff968605fde8557b2caf4a3fd94ba40b54f74eba39558fb7dbe6103f"
sourceCorrections: []
---

当我们训练一个神经网络 (neural network)来估算Q值时，更新过程中存在一个潜在的不稳定性问题。Q学习中更新的基本设想，正是它激发了DQN的损失函数 (loss function)，其目标是迭代地调整Q值使其更接近某个目标值。我们希望由权重 (weight) $\theta$ 参数 (parameter)化的Q网络能够预测一个值 $Q(S_t, A_t; \theta)$，使其更接近目标值，目标值通常这样计算：


$$
y_t \approx R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)
$$


这里请注意一个重要之处：权重 $\theta$ 出现在这个隐式方程的*两边*。我们正在调整 $\theta$，以便使 $Q(S_t, A_t; \theta)$ 与一个也依赖于 $\theta$ 的目标匹配。

### 移动目标的难题

试想您正尝试学习到一个特定点的距离，但每当您进行一次测量并调整您的估计时，该点本身就会根据您的调整而略微移动。这使得收敛变得困难。类似地，在带有神经网络 (neural network)的标准Q学习中，每个梯度更新步骤都会改变权重 (weight) $\theta$。这反过来又会改变下一个状态 $S_{t+1}$ 的Q值，从而在网络收敛到先前目标之前，有效地转移了目标 $y_t$。

这种“移动目标”问题可能在训练过程中导致振荡和发散。网络可能难以确定准确的Q值，因为它试图预测的值会因学习过程本身而持续变化。这种不稳定性是早期将深度学习 (deep learning)与强化学习 (reinforcement learning)结合时的一个重要障碍。

### 引入目标网络

为解决这种不稳定性，DQN算法引入了一种巧妙的方法：使用一个单独的**目标网络**。主要思想是使用一组*固定*的参数 (parameter)来计算目标值 $y_t$，这些参数不会立即更新。

它的工作方式如下：

1. 我们维护两个神经网络 (neural network)：

   - **在线网络**（或策略网络），带有权重 (weight) $\theta$。这是我们通过梯度下降 (gradient descent)积极训练的网络。它根据当前策略（例如，使用其Q值的epsilon-greedy）决定要采取的行动。
   - **目标网络**，带有权重 $\theta^-$。这个网络与在线网络具有*完全相同的架构*。
2. 当计算损失函数 (loss function)的TD目标时，我们使用目标网络：

   
   $$
   y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)
   $$
   

   重要的是，这里使用的权重 $\theta^-$ 在一段时间内是*保持不变的*。
3. 然后使用在线网络的预测和这个稳定的目标来计算损失函数：

   
   $$
   L(\theta) = \mathbb{E}_{(S_t, A_t, R_{t+1}, S_{t+1}) \sim D} \left[ \left( y_t - Q(S_t, A_t; \theta) \right)^2 \right]
   $$
   
   
   $$
   L(\theta) = \mathbb{E}_{(S_t, A_t, R_{t+1}, S_{t+1}) \sim D} \left[ \left( (R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)) - Q(S_t, A_t; \theta) \right)^2 \right]
   $$
   
4. 只有在线网络的权重 $\theta$ 会根据损失 $L(\theta)$ 的梯度进行更新。目标网络的权重 $\theta^-$ *不会*被这个梯度步骤修改。

这使得目标计算与正在积极更新的权重分离，从而提供一个更稳定的学习信号。

> 此图显示了使用在线网络和目标网络的DQN训练过程。目标网络($\theta^-$)为损失计算提供稳定的目标，而在线网络($\theta$)通过优化进行更新。权重会定期从在线网络复制到目标网络。

### 更新目标网络权重 (weight)

如果目标网络的权重 $\theta^-$ 从未改变，在线网络 $\theta$ 将会追逐一个越来越过时的目标。因此，我们需要定期更新 $\theta^-$，以反映在线网络所取得的进展。

一种常用且有效的策略是执行**硬更新**：
每隔 $C$ 个训练步骤（其中 $C$ 是一个超参数 (parameter) (hyperparameter)，通常为数百或数千步），将权重从在线网络复制到目标网络：


$$
\theta^- \leftarrow \theta
$$


在这些更新之间，$\theta^-$ 保持固定。这种定期更新确保目标网络缓慢跟踪已学习的策略，提供稳定性，而不会完全静止。$C$ 的选择涉及一种权衡：较小的 $C$ 使目标更快地跟踪在线网络（可能重新引入一些不稳定性），而较大的 $C$ 提供更高的稳定性，但如果目标变得过于过时，可能会减慢学习速度。

*注：另一种方法是“软更新”或波利亚克平均，其中目标权重在每一步都轻微更新：$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$，其中 $\tau \ll 1$。这提供了更平滑的更新，但增加了一个额外的超参数 $\tau$。为求简单，我们通常从硬更新开始。*

### 益处以及与经验回放的配合

使用目标网络提供以下几点益处：

- **增加稳定性：** 通过固定目标在一定数量的更新周期内，它显著减少了在朴素Q学习与函数逼近中出现的振荡和发散。
- **改善收敛：** 稳定的目标使得网络更容易收敛到更好的Q值估计。
- **解耦更新：** 它打破了有害的关联，即一次更新会立即影响它自己的目标计算。

目标网络与经验回放配合使用效果良好。经验回放打破了从环境中抽取到的连续样本之间的时间关联，而目标网络则打破了网络当前估计 $Q(S_t, A_t; \theta)$ 与目标值 $y_t$ 之间的关联。这两种方法（经验回放和目标网络）共同对原始DQN算法的成功起到了非常重要的作用，并且在许多深度强化学习 (reinforcement learning)实现中仍然是标准实践。它们为深度神经网络 (neural network)有效学习强化信号提供了所需的稳定性。

## 参考资料

- [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Andras Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍深度Q网络（DQN）的原始论文，详细阐述了使用目标网络和经验回放来稳定深度强化学习。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本广受认可的教科书，全面回顾了强化学习，包括Q学习和函数逼近的考量。
- [Lecture 6: Value Function Approximation](https://www.davidsilver.uk/wp-content/uploads/2020/03/Value-Function-Approximation.pdf) — David Silver (2015)
  Journal: UCL Course on Reinforcement Learning; Publisher: UCL
  来自知名学术系列讲座之一，解释了强化学习中的价值函数逼近，包括目标网络在DQN中的作用。
