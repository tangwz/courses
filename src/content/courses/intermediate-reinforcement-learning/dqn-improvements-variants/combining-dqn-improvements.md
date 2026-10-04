---
course: "intermediate-reinforcement-learning"
chapter: "dqn-improvements-variants"
lesson: "combining-dqn-improvements"
sourceId: 5126
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/combining-dqn-improvements"
title: "DQN改进的结合"
description: "整合DQN多项增强（如DDQN和对偶网络）的策略。"
order: 4
plots: []
sourceHash: "2c98bc3787fa37d87aa2bf9b79e316dabf4bd0fb1cc842490570af19ff1fe6c7"
sourceCorrections: []
---

双重DQN（DDQN）可以减少价值高估，而对偶网络架构则能更准确地估计状态价值和动作优势。一个很自然的问题是：这些改进能一起使用吗？幸运的是，答案是肯定的。这些技术处理DQN训练过程的不同方面，并且很大程度上是互补的。将它们结合起来，与单独使用任何一项增强相比，通常能显著提升性能和稳定性。

### 整合双重DQN和对偶网络

核心思想是同时使用对偶架构用于在线网络（用于动作选择和梯度计算）和目标网络（用于计算目标Q值），同时采用双重DQN的更新规则来计算该目标值。

回顾对偶网络架构计算Q值的方式如下：


$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$


这里，$\theta$ 表示价值流 ($V$) 和优势流 ($A$) 之间共享的参数 (parameter)，而 $\beta$ 和 $\alpha$ 分别是价值流和优势流特有的参数。我们用 $(\theta, \alpha, \beta)$ 表示在线网络的参数，用 $(\theta^-, \alpha^-, \beta^-)$ 表示目标网络的参数。

现在，我们来引入双重DQN的原则。DDQN目标 $Y_t$ 的计算方式如下：


$$
Y_t^{DDQN} = R_{t+1} + \gamma Q_{\text{目标}}(S_{t+1}, \arg\max_{a'} Q_{\text{在线}}(S_{t+1}, a'; \theta, \alpha, \beta); \theta^-, \alpha^-, \beta^-)
$$


将它们结合起来：

1. **动作选择**：使用带有对偶架构的*在线网络*来确定下一个状态 $S_{t+1}$ 的最佳动作 $a^*$：

   
   $$
   a^* = \arg\max_{a'} Q_{\text{在线}}(S_{t+1}, a'; \theta, \alpha, \beta)
   $$
   

   具体来说，您会使用在线网络的优势流计算所有动作 $a'$ 的优势 $A(S_{t+1}, a'; \theta, \alpha)$，并选择对应最大优势的动作 $a^*$（因为对于给定状态，$V(S_{t+1})$ 项在所有动作中是恒定的）。
2. **动作评估**：使用*目标网络*（同样带有对偶架构）来评估在状态 $S_{t+1}$ 中执行动作 $a^*$ 的Q值：

   
   $$
   Q_{\text{目标}}(S_{t+1}, a^*; \theta^-, \alpha^-, \beta^-) = V_{\text{目标}}(S_{t+1}; \theta^-, \beta^-) + \left( A_{\text{目标}}(S_{t+1}, a^*; \theta^-, \alpha^-) - \frac{1}{|\mathcal{A}|} \sum_{a''} A_{\text{目标}}(S_{t+1}, a''; \theta^-, \alpha^-) \right)
   $$
   
3. **目标计算**：使用奖励 $R_{t+1}$ 和从目标网络评估出的Q值来构成最终的目标值 $Y_t$：

   
   $$
   Y_t = R_{t+1} + \gamma Q_{\text{目标}}(S_{t+1}, a^*; \theta^-, \alpha^-, \beta^-)
   $$
   
4. **损失计算**：计算目标 $Y_t$ 与在线网络为原始状态-动作对 $(S_t, A_t)$ 预测的Q值之间的损失（例如，均方误差或Huber损失）：

   
   $$
   \text{损失} = \mathcal{L}(Y_t - Q_{\text{在线}}(S_t, A_t; \theta, \alpha, \beta))
   $$
   
5. **梯度更新**：使用从该损失中导出的梯度来更新在线网络的参数 $(\theta, \alpha, \beta)$。定期通过复制在线网络参数来更新目标网络参数 $(\theta^-, \alpha^-, \beta^-)$。

这种结合方法，通常被称为对偶DDQN，兼具双重DQN减少高估偏差的优点和对偶网络架构提升的特征学习能力。

> 该图说明了对偶双重DQN中目标值 ($Y_t$) 的数据流。动作选择使用在线网络，而动作评估使用目标网络，两者都采用了对偶架构。

### 更多改进：优先经验回放（PER）

正如之前简单提及，优先经验回放（PER）是另一个重要的改进，可以在对偶DDQN之上叠加。PER不是从回放缓冲区中均匀采样转换，而是根据TD误差来采样转换。代理预测高度不准确（TD误差大）的转换被认为是更“令人意外”或更有信息量的，因此会被更频繁地回放。

整合PER包括：

1. **存储TD误差**：当一个转换 $(S_t, A_t, R_{t+1}, S_{t+1})$ 被添加到回放缓冲区时，计算其初始TD误差并与该转换一起存储。
2. **优先采样**：采样一个小批量时，根据这些存储的TD误差（或从中导出的排名）分配概率。误差越大意味着被采样的概率越高。
3. **重要性采样权重 (weight)**：为了纠正非均匀采样引入的偏差，为小批量中的每个转换计算重要性采样（IS）权重。这些权重会调整梯度更新，降低频繁采样转换的更新权重。
4. **更新优先级**：在对采样到的转换执行梯度更新后，使用新计算的TD误差更新其在回放缓冲区中的优先级。

将对偶DDQN与PER结合，常能带来许多离散动作空间任务的当前最佳性能，例如在街机学习环境（ALE）中发现的任务。实现复杂度会增加，但在样本效率和最终性能方面的潜在提升可能是可观的。

总结来说，这些DQN改进的模块化特性使得它们能够有效结合。从基础DQN开始，加入双重DQN可以解决高估问题，对偶网络改善了价值函数的表示，优先经验回放则将学习重点放在信息量最大的转换上。实施这些组合提供了一个强大的工具集，用于解决复杂的强化学习 (reinforcement learning)问题。

## 参考资料

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — Hado van Hasselt, Arthur Guez, David Silver (2016)
  Journal: AAAI Conference on Artificial Intelligence; Publisher: AAAI; DOI: [10.48550/arXiv.1509.06461](https://doi.org/10.48550/arXiv.1509.06461)
  提出双Q学习算法以减少Q学习中价值高估偏差的原始论文。
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) — Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas (2016)
  Journal: International Conference on Machine Learning (ICML); Pages: 1995-2003; DOI: [10.48550/arXiv.1511.06581](https://doi.org/10.48550/arXiv.1511.06581)
  介绍了深度强化学习的Dueling网络架构，分离了状态值和优势函数。
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Tom Schaul, John Quan, Ioannis Antonoglou, David Silver (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1511.05952](https://doi.org/10.48550/arXiv.1511.05952)
  介绍了优先经验回放，这是一种根据时间差分误差采样转换的经验回放机制。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍深度Q网络（DQN）的开创性论文，用于直接从原始像素输入玩Atari游戏。
