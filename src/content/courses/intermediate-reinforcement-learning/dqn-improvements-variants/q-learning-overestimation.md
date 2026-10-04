---
course: "intermediate-reinforcement-learning"
chapter: "dqn-improvements-variants"
lesson: "q-learning-overestimation"
sourceId: 5117
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/q-learning-overestimation"
title: "Q-学习中的估值过高问题"
description: "了解标准Q-学习和DQN如何高估动作价值。"
order: 1
plots: ["plots/5117-0.json"]
sourceHash: "9157354aa1573e915c65bbca06bf23e131c976517eab5b5201075fd9b76546bf"
sourceCorrections: []
---

标准深度Q网络代表着一个重要进展，使我们能够将强化学习 (reinforcement learning)应用于高维状态空间的问题。然而，底层的Q-学习机制，尤其是其更新规则，带有一种固有的乐观倾向：它会系统性地高估动作价值。这不仅仅是一个微小的不准确；它会对学习性能和稳定性产生负面影响。我们来分析一下原因。

回顾表格方法中使用的标准Q-学习更新规则：


$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$


核心部分是$\max_{a'} Q(s', a')$项。这一部分估算可从下一状态$s'$获得的最大可能价值。问题出现是因为$Q(s', a')$本身就是*估计值*，特别是在训练早期或使用函数近似器（如神经网络 (neural network)）时。这些估计值不可避免地包含噪声或误差。

思考一下当你对多个带噪声的估计值取最大值时会发生什么。如果某些估计值随机地高于其真实值，而另一些则随机地低于其真实值，那么$\max$操作更有可能选择其中一个高估的值。它不会平均误差；它会主动选择它所看到的最大的值，可能会放大正向噪声。这会导致下一状态估计值中存在持续的正向偏差。

这种现象被称为**最大化偏差**。设想在状态$s'$中你有几个可用的动作，你的当前$Q$函数估算了它们的值，这些值包含一些随机误差：



![对带噪声Q值估计应用Max操作](plots/5117-0.json)



> 真实的最佳动作是动作2，其值为1.5。然而，由于估计误差，智能体对动作2的估计值为1.9。$\max$操作选择了这个被高估的值（1.9）用于目标计算，导致与真实最大值（1.5）相比存在正向偏差。

在深度Q网络中，这个问题依然存在。用于训练网络的目标值$y_i$计算如下：


$$
y_i = r_i + \gamma \max_{a'} Q_{target}(s'_{i}, a'; \theta^{-})
$$


即使我们使用带有参数 (parameter)$\theta^{-}$的单独目标网络$Q_{target}$以稳定训练，$\max$操作仍然应用于该目标网络产生的估计值。如果这些目标Q值有噪声或不确定，最大化步骤将继续引入一个向上的偏差到目标值$y_i$中。

这种高估会带来什么后果？

1. **次优策略：** 智能体可能会学会偏好导致Q值被明显高估的状态，而不是真正最优的状态的动作。
2. **收敛减慢：** 这种偏差会使学习过程稳定性降低，并可能更慢，因为网络试图匹配这些膨胀的目标值。
3. **不稳定：** 在某些情况下，严重的高估可能导致训练期间的震荡或发散。

理解这种最大化偏差很重要，因为它促使了对基本DQN算法的几项改进。已经开发出专门用于缓解这种过高估计问题的技术，从而实现更可靠、更高效的学习。最直接的解决方案之一是双DQN（Double DQN），它修改了目标值的计算方式。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  全面教材，解释了Q学习、时序差分学习、最大化偏差问题以及作为解决方案的双Q学习。
- [Double Q-learning](https://proceedings.neurips.cc/paper/2010/file/091d584fced50280f63ee87d37ee36f2-Paper.pdf) — Hado van Hasselt (2010)
  Journal: Advances in Neural Information Processing Systems (NIPS); Publisher: Neural Information Processing Systems Foundation; Volume: 23; Pages: 2613-2621
  介绍了双Q学习，以减轻强化学习中的最大化偏差，并提供了理论分析。
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461) — Hado van Hasselt, Arthur Guez, and David Silver (2016)
  Journal: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI); Pages: 2094-2100
  将双Q学习扩展到深度Q网络（DQN），证明了其在减少过高估计和提高性能方面的有效性。
