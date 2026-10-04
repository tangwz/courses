# 双重深度Q网络 (DDQN)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/double-dqn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

深度Q网络（DQN）将Q学习与深度神经网络 (neural network)，以及经验回放和目标网络等技术结合，使智能体能够在高维状态空间中高效学习。然而，DQN中使用的标准Q学习更新可能会遇到一个主要问题：Q值过高估计。

### 过高估计偏差问题

回顾标准DQN中，针对转移$(S_t, A_t, R_{t+1}, S_{t+1})$的目标值($Y_t$)计算：


$$
Y_t^{DQN} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta_t^-)
$$


这里，$\theta_t^-$代表目标网络的参数 (parameter)。在线网络参数$\theta_t$的更新旨在最小化$Q(S_t, A_t; \theta_t)$与此目标$Y_t^{DQN}$之间的损失。

问题在于目标计算中的$\max_{a'}$操作。目标网络估算的Q值（$Q(S_{t+1}, a'; \theta_t^-)$）本身是对真实动作值的带有噪声的近似。当我们从这些带有噪声的估计中取最大值时，我们更有可能选择一个被高估的值，而不是一个被低估的值。设想有几个动作，它们的真实值相近；由于噪声，有些估计会高于其他估计。`max`操作符将持续选择这些较高、可能被高估的值。这种系统性的正偏差会在学习过程中传播，导致过于乐观的值估计、训练期间的不稳定性以及潜在的次优策略。

### 使用双重DQN分离选择与评估

双重深度Q网络（DDQN），由Hado van Hasselt, Julià Ghesu, Matej Horgan, Maurice Wiering和David Silver提出，通过在计算目标时，将最优动作的*选择*与该动作值的*评估*分离开来，直接解决了这个过高估计问题。

DDQN不使用目标网络$\theta_t^-$同时进行最大化动作的*选择*和Q值的*评估*，而是使用*在线*网络$\theta_t$来为下一个状态$S_{t+1}$选择最优动作，然后使用*目标*网络$\theta_t^-$来评估该特定所选动作的Q值。

DDQN中的目标值计算变为：


$$
Y_t^{DDQN} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta_t); \theta_t^-)
$$


让我们分解一下：

1. **动作选择:** 首先，我们使用当前在线网络$Q(S_{t+1}, a'; \theta_t)$来找到它*认为*在下一个状态$S_{t+1}$中能使Q值最大化的动作$a^*$。这便是$\arg\max_{a'} Q(S_{t+1}, a'; \theta_t)$部分。
2. **动作评估:** 然后，我们不直接使用可能被高估的值$Q(S_{t+1}, a^*; \theta_t^-)$（DQN会通过对目标网络值取最大值来隐式执行），而是使用目标网络$Q(.; \theta_t^-)$来获取在线网络选择的*那个特定动作* $a^*$的Q值估计。这便是$Q(S_{t+1}, a^*; \theta_t^-)$部分，其中$a^*$是步骤1中找到的动作。

> 标准DQN和双重DQN（DDQN）中目标值计算流程的比较。DDQN使用在线网络进行动作选择（argmax），并使用目标网络评估所选动作的值。

### 为什么这有帮助？

在线网络（$\theta_t$）和目标网络（$\theta_t^-$）是不同的参数 (parameter)集（目标网络通常是在线网络的定期更新副本）。虽然两个网络都可能对某些动作存在噪声和潜在的过高估计，但*两个*网络同时高估*同一*次优动作值的可能性较小。

如果在线网络选择动作$a^*$是因为其估计$Q(S_{t+1}, a^*; \theta_t)$当前被高估，那么目标网络对该同一动作的估计$Q(S_{t+1}, a^*; \theta_t^-)$可能更接近真实值（或者至少，高估程度较轻）。通过使用目标网络对在线网络所选动作的值，DDQN减少了将最大可能过高估计传播到目标值$Y_t$中的机会。这会带来更保守和更准确的值估计。

### 实现与优势

如果你已经有一个DQN实现，那么DDQN的实现非常直接。唯一需要的更改是修改训练循环中目标值$Y_t$的计算方式。你仍然需要经验回放以及独立的在线网络和目标网络。

DDQN的主要优势是：

1. **过高估计减少:** 与标准DQN相比，它明显地带来更低（且通常更准确）的Q值估计。
2. **稳定性提升:** 通过减少过高估计偏差，训练通常变得更加稳定。
3. **性能更好:** 在许多环境中，DDQN比标准DQN取得了更好的最终策略表现。

因为修改简单，并且优势在许多领域都显著且一致，DDQN被认为是原始DQN算法的一种标准改进，并常被用作默认选择或基线。它代表了完善基于值的方法在深度强化学习 (reinforcement learning)中的重要一步。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本关于强化学习的著作，涵盖Q-学习、价值函数估计及相关算法。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Andras G. P. Szepesvari, Helen King, Raia Hadsell and Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了基础的深度Q网络 (DQN) 算法，它通过深度神经网络和经验回放将Q学习应用于高维状态空间。
- [Double Q-learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEI3wln1HUXlvY2h9BfPboY30OjG0Wm-gIhd3ljtLnOU2DhB1QBVbC2SlXiqCvNwWbIkV9MZQYF2nsg_MJFxNCagtyxz0SrUrmVpPnhsPLA6p3PDsqlobTjiu9zg_UuLJfrXGMM1IpzSe39zd9OfA==) — Hado van Hasselt (2010)
  Journal: Advances in Neural Information Processing Systems 23 (NIPS 2010); Publisher: Curran Associates, Inc.; Volume: 23; Pages: 2613-2621
  介绍了双Q学习算法，通过使用独立的价值估计进行动作选择和评估，减少了过高估计偏差。
- [Deep Reinforcement Learning with Double Q-learning](https://doi.org/10.48550/arXiv.1509.06461) — Hado van Hasselt, Arthur Guez and David Silver (2016)
  Journal: AAAI Conference on Artificial Intelligence (AAAI); DOI: [10.48550/arXiv.1509.06461](https://doi.org/10.48550/arXiv.1509.06461)
  提出了双深度Q网络 (DDQN)，是双Q学习在深度强化学习中的扩展，专门解决了DQN中存在的过高估计偏差。这是本节的主要参考文献。

---

[上一节](04-%E7%94%A8%E4%BA%8E%E8%AE%AD%E7%BB%83%E7%A8%B3%E5%AE%9A%E6%80%A7%E7%9A%84%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C.md) · [下一节](06-%E5%AF%B9%E5%81%B6%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
