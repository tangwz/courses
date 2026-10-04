---
course: "intermediate-reinforcement-learning"
chapter: "actor-critic-methods"
lesson: "asynchronous-actor-critic-a3c"
sourceId: 5162
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/asynchronous-actor-critic-a3c"
title: "异步优势参与者-评价者算法 (A3C)"
description: "使用多个并行参与者收集多样的经验并提高效率。"
order: 4
plots: []
sourceHash: "b8ac5a8e21bbc0976d22aa98ddd3fd58b51d1cef1447bf0e5b6fe7b7cacfad27"
sourceCorrections: []
---

虽然优势参与者-评价者算法 (A2C) 通过使用优势函数指导策略更新，提供了扎实的框架，但其同步特性有时可能成为瓶颈。A2C中的梯度通常从多个并行运行的参与者收集的经验批次中汇总，然后取平均值，并应用于更新中央模型。这种同步步骤可能会限制吞吐量 (throughput)，特别是当某些参与者比其他参与者慢时。此外，训练深度强化学习 (reinforcement learning)模型通常受益于多样的经验，以防止过拟合 (overfitting)并提升稳定性。

A3C（异步优势参与者-评价者算法）就此出现，这是一种旨在通过并行和异步处理这些问题的变体。A3C由DeepMind的研究人员开发，因其高效和出色的性能而影响深远，特别是在需要多核CPU的任务上。

### 异步的优势

A3C的核心思想简单而有效：A3C不是同步协调多个参与者，而是并行运行多个独立的参与者-学习者代理，每个代理都有自己的环境副本和参与者与评价者网络的本地副本。

其工作方式如下：

1. **全局网络：** 存在一个中央全局网络，包含参数 (parameter)（参与者的 $\theta$，评价者的 $\phi$），所有代理最终都会促进其改进。
2. **工作代理：** 创建多个“工作”进程（或线程）。每个工作代理都具有：
   - 参与者和评价者网络的本地副本，最初与全局网络的参数同步。
   - 自己的环境实例。
3. **并行交互：** 每个工作代理独立地与其环境副本交互一定数量的步长（或直到达到终止状态），收集状态、动作、奖励和下一状态 ($s, a, r, s'$)。
4. **局部梯度计算：** 使用其收集到的经验，每个工作代理计算其本地参与者网络（基于策略梯度目标，可能使用优势估计）和其本地评价者网络（基于最小化价值函数误差，例如使用TD误差）的梯度。优势计算 $A(s, a) = Q(s, a) - V(s)$ 通常使用N步回报或广义优势估计（GAE）进行估计，类似于A2C，但由工作代理在本地计算。
5. **异步更新：** 重要的是，一旦工作代理计算出其梯度，它会直接发送这些梯度以更新**全局网络参数**。此更新异步发生，这意味着工作代理不等待其他工作代理。它发送其更新并立即将其本地网络参数重置以匹配（可能已更新的）全局参数，然后继续与其环境交互。

> 一张展示A3C架构的图表。多个工作代理与独立的环境实例交互，在本地计算梯度，并异步更新一个中央全局网络。工作代理会定期将其本地网络参数与全局网络参数同步。

### 异步的优点

这种异步结构具有几个显著优点：

1. **去相关经验：** 由于每个工作代理使用略微不同的策略版本（因为全局网络在不断更新）与其自己的环境实例交互，因此在各个工作代理之间收集的经验随着时间的推移更加多样且相关性更低。这种固有的多样性作用类似于DQN中使用的经验回放缓冲区，稳定了学习过程，而无需存储大量的历史数据。许多A3C实现根本不使用回放缓冲区。
2. **提高效率：** 在多核CPU系统上，A3C可以非常高效。当一个工作代理正在计算梯度或等待一个环境步长时，其他工作代理可以执行计算或发送更新。与单线程算法或可能需要等待最慢工作代理的同步方法相比，这种并行性带来了更高的数据吞吐量 (throughput)。
3. **鲁棒性：** 整个训练过程对任何单个工作代理的性能不那么敏感。

### 实现方面

实现A3C时，有几个细节是常见的：

- **共享优化器：** 所有工作代理的梯度通常通过共享优化器（例如RMSProp或Adam）用于更新全局网络参数 (parameter)。
- **参数共享：** 通常，参与者和评价者神经网络 (neural network)的较低层是共享的，而策略分布（参与者）和价值估计（评价者）则有独立的输出头。这可以提高学习效率，使两个组件都能受益于共同的特征提取。
- **熵正则化 (regularization)：** 与A2C一样，在策略梯度目标中添加熵奖惩是标准做法。这通过惩罚过快变得过于确定性的策略来鼓励多样化。参与者的损失可能看起来像这样：
  
  $$
  L_{actor} = -\log \pi_\theta(a_t|s_t) A(s_t, a_t) - \beta H(\pi_\theta(\cdot|s_t))
  $$
  
  其中 $A(s_t, a_t)$ 是优势估计，$H$ 是策略熵，$\beta$ 是一个控制熵奖惩强度的超参数 (hyperparameter)。
- **N步回报：** 工作代理通常根据N步回报计算梯度，以估计优势，平衡偏差和方差。例如，收集 $n$ 步经验 $(s_t, a_t, r_{t+1}, ..., s_{t+n-1}, a_{t+n-1}, r_{t+n}, s_{t+n})$ 可以计算N步回报：
  
  $$
  R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n V_\phi(s_{t+n})
  $$
  
  优势可以估计为 $A(s_t, a_t) \approx R_t^{(n)} - V_\phi(s_t)$。

### A3C 对比 A2C

虽然A3C引入了强大的异步更新方案，但其同步对应物A2C通常更简单实现，并且在使用GPU时可能更有效。A2C通常批量处理来自多个并行参与者的经验，并执行一次大的梯度更新。这种批量处理可以比A3C频繁、小规模的异步更新更有效地利用GPU并行性，A3C在使用GPU时常因通信开销而成为瓶颈。

然而，A3C的优势在于其有效使用多核CPU的能力，以及通过多样化异步过程实现的固有稳定性。它表明可以实现高效、稳定的深度强化学习 (reinforcement learning)，而无需依赖经验回放缓冲区。

总而言之，A3C是一个里程碑式的参与者-评价者算法，它使用异步并行代理来提高训练效率和稳定性。通过让多个工作代理独立地遍历状态空间的不同区域并异步更新全局网络，A3C有效地去除了经验的相关性并加速了学习，特别是在基于CPU的系统上。它代表着在将参与者-评价者方法扩展到复杂问题方面迈出了重要一步。

## 参考资料

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: ICML 2016; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍A3C算法的原始论文，详细阐述其异步架构及对高效深度强化学习的优势。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习的标准教科书，提供了Actor-Critic方法、n步回报和策略梯度的基础概念。
- [Reinforcement Learning (UCL Course)](https://www.davidsilver.uk/teaching/) — David Silver (2015)
  一系列讲座，清晰地阐述了强化学习算法，包括Actor-Critic方法的详细说明。
