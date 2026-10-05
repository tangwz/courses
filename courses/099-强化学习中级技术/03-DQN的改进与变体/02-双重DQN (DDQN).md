# 双重DQN (DDQN)

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/double-dqn-ddqn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准Q学习及其深度学习 (deep learning)扩展DQN，存在高估动作价值的倾向。这是因为其更新规则涉及对可能存在噪声或不准确的Q值估计进行最大化操作。标准DQN的目标值计算如下：


$$
y_t^{DQN} = r_t + \gamma \max_{a'} Q_{\theta'}(s_{t+1}, a')
$$


在这里，目标网络 $\theta'$ 用于两个目的：首先，选择在下一状态 $s_{t+1}$ 中被认为能产生最高Q值的动作 $a'$；其次，评估所选动作的Q值。如果目标网络恰好高估了*任何*动作 $a'$ 的价值，则 $\max$ 操作符很可能会选择那个被高估的值，从而导致目标值 $y_t^{DQN}$ 出现正向偏差。这种持续的高估会通过学习过程传播，可能导致次优策略收敛和不稳定。

双重DQN（DDQN），由Hado van Hasselt、Arthur Guez和David Silver于2015年提出，提供了一种巧妙的方法来缓解这种高估偏差。它的核心思想是将最佳下一动作的*选择*与其价值的*评估*分离。DDQN不使用同一个网络（目标网络）来完成这两项任务，而是使用两个不同的网络。

### 双重DQN的更新方式

具体而言，DDQN如下修改了目标值的计算：

1. **动作选择：** 使用*当前在线网络* $Q_{\theta}$（正在积极训练的网络）来确定它在下一状态 $s_{t+1}$ 中估计的最佳动作 $a^*$。
   
   $$
   a^* = \arg\max_{a'} Q_{\theta}(s_{t+1}, a')
   $$
   
2. **动作评估：** 使用*目标网络* $Q_{\theta'}$（定期更新的稳定网络）来评估上一步中选定的动作 $a^*$ 的Q值。
   
   $$
   Q_{\theta'}(s_{t+1}, a^*)
   $$
   
3. **DDQN目标值：** 将这些结合起来形成新的目标值 $y_t^{DDQN}$。
   
   $$
   y_t^{DDQN} = r_t + \gamma Q_{\theta'}(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a'))
   $$
   

将此与标准DQN目标值仔细比较。在DQN中，`argmax` 和价值估计都完全依赖于目标网络 $Q_{\theta'}$。在DDQN中，`argmax` 使用在线网络 $Q_{\theta}$，而价值估计则使用目标网络 $Q_{\theta'}$。

### 为何分离有助于改进？

在线网络 $\theta$ 和目标网络 $\theta'$ 表示不同的参数 (parameter)集合（回想一下，$\theta'$ 通常是 $\theta$ 的延迟副本）。尽管两个网络都可能产生估计误差，但两个网络同时高估*同一*动作价值的可能性较小。

通过使用在线网络选择动作 ($a^* = \arg\max_{a'} Q_{\theta}(s_{t+1}, a')$)，我们仍然选取*当前*策略认为最佳的动作。然而，随后使用目标网络评估*该特定动作*的价值 ($Q_{\theta'}(s_{t+1}, a^*)$)，我们得到了一个偏差可能更小的估计。如果在线网络错误地选择了一个被 $Q_{\theta}$ 高估但未被 $Q_{\theta'}$ 明显高估的动作，那么产生的目标值 $y_t^{DDQN}$ 将比 $y_t^{DQN}$ 的膨胀程度更小。这有助于打破高估的正向反馈循环。

下面的图表说明了DQN与DDQN在目标值计算方式上的差异。

> 目标值计算流程的比较。DQN仅使用目标网络来选择最大价值动作并评估其价值。DDQN使用在线网络选择动作，并使用目标网络评估所选动作。

### 实现影响

实现DDQN仅需对标准DQN的实现进行少量修改。您无需计算下一状态的 `max(target_q_values)`，而是需要：

1. 对下一状态 $s_{t+1}$ 通过*在线*网络进行一次前向传播，以获取所有 $a'$ 的 $Q_{\theta}(s_{t+1}, a')$。
2. 使用 `argmax` 找出使这些在线Q值最大的动作 $a^*$。
3. 对下一状态 $s_{t+1}$ 通过*目标*网络进行一次前向传播，以获取所有 $a'$ 的 $Q_{\theta'}(s_{t+1}, a')$。
4. 选择与第2步中找到的动作 $a^*$ 对应的目标Q值。
5. 在贝尔曼更新计算中 ($r_t + \gamma Q_{\theta'}(s_{t+1}, a^*)$) 使用这些选定的目标Q值。

DQN的其余机制，包括经验回放和目标网络的定期更新，保持不变。DDQN增加的计算开销通常可以忽略不计，因为它主要是在目标值计算步骤中额外进行一次在线网络的前向传播，这通常比梯度更新的反向传播 (backpropagation)成本低得多。

通过降低高估偏差，DDQN通常能带来更稳定的训练，并且与原始DQN算法相比，可以收敛到更好的策略，使其成为深度强化学习 (reinforcement learning)中一项重要且被广泛使用的改进。

## 参考资料

- [Deep Reinforcement Learning with Double Q-learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295) — Hado van Hasselt, Arthur Guez, David Silver (2016)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 30; Pages: 2094-2100; DOI: [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)
  这篇论文介绍了双深度Q网络（DDQN）算法，旨在解决深度Q网络中的过高估计偏差问题。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本关于强化学习的综合教科书，涵盖了Q学习、DQN及其改进算法。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller (2015)
  Journal: Nature; Publisher: Springer Nature; Volume: 518; Pages: 529–533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇原始论文介绍了深度Q网络（DQN），为双深度Q网络提供了基础背景。

---

[上一节](01-Q-%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E4%BC%B0%E5%80%BC%E8%BF%87%E9%AB%98%E9%97%AE%E9%A2%98.md) · [下一节](03-%E5%AF%B9%E5%81%B6%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
