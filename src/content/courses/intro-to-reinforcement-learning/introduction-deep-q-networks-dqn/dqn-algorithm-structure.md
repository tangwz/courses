---
course: "intro-to-reinforcement-learning"
chapter: "introduction-deep-q-networks-dqn"
lesson: "dqn-algorithm-structure"
sourceId: 1662
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/dqn-algorithm-structure"
title: "DQN 算法结构"
description: "概述了包含经验回放和目标网络的完整深度 Q-网络算法。"
order: 5
plots: []
sourceHash: "738b1ec24a945e4ed4411b8cf9ba98a1b96b2324285ae618ece23b2ab42993b4"
sourceCorrections: []
---

深度 Q-网络 (DQN) 算法将深度神经网络 (neural network)与 Q-学习相结合。它采用经验回放和目标网络来提高训练稳定性。该标准算法提供了一个框架，用于训练智能体在复杂环境中学习最优动作值函数。

核心思想依然源于 Q-学习：我们希望学习一个动作值函数 $Q(s, a)$，它近似表示在状态 $s$ 执行动作 $a$ 后并遵循最优策略的预期回报。然而，我们不使用表格，而是使用一个由权重 (weight) $\theta$ 参数 (parameter)化的深度神经网络来表示这个函数：$Q(s, a; \theta)$。为了处理前面讨论的不稳定性问题，我们引入了两项主要改进：

1. **经验回放：** 我们将智能体的经验（状态、动作、奖励、下一状态元组）存储在一个经验回放缓冲区中。在训练期间，我们从缓冲区中随机抽取这些经验的小批量，用于更新网络。这打破了观测序列中的时间关联，并平滑了数据分布的变化。
2. **目标网络：** 我们使用一个独立的神经网络，称为目标网络，其权重为 $\theta^-$，用于计算更新规则中使用的目标 Q 值。权重 $\theta^-$ 会定期从主 Q 网络权重 $\theta$ 复制而来，但在两次更新之间会保持固定若干步。这为 Q-学习更新提供了一个更稳定的目标，防止目标值追逐主网络持续变化的预测。

### DQN 算法流程

下面是典型 DQN 训练循环的步骤分解：

1. **初始化：**

   - 初始化具有固定容量的经验回放缓冲区 $D$。
   - 使用随机权重 (weight) $\theta$ 初始化主 Q 网络 $Q$。
   - 使用权重 $\theta^- = \theta$ 初始化目标 Q 网络 $\hat{Q}$。
2. **幕循环：** 对于每一幕：

   - 重置环境并获取初始状态 $s_1$。
   - 如有必要，对初始状态进行预处理（例如，堆叠帧）以创建网络的输入表示。
3. **步循环：** 在本幕中的每一步 $t = 1, 2, ... T$：

   - **动作选择：** 使用源自主 Q 网络的 $\epsilon$-贪婪策略，根据当前状态 $s_t$ 选择动作 $a_t$：
     - 以 $\epsilon$ 的概率，选择一个随机动作。
     - 以 $1-\epsilon$ 的概率，选择 $a_t = \arg\max_a Q(s_t, a; \theta)$。
       （通常，$\epsilon$ 会逐渐衰减，从高值开始并随时间降低，以从（行为）随机性转向（行为）利用性）。
   - **执行动作：** 在环境中执行动作 $a_t$。观察奖励 $r_{t+1}$ 和下一状态 $s_{t+1}$。预处理 $s_{t+1}$。
   - **存储经验：** 将转换元组 $(s_t, a_t, r_{t+1}, s_{t+1})$ 存储在经验回放缓冲区 $D$ 中。
   - **采样小批量：** 如果经验回放缓冲区包含足够的经验，从 $D$ 中随机采样 $N$ 个转换 $(s_j, a_j, r_{j+1}, s_{j+1})$ 的小批量。
   - **计算目标值：** 对于小批量中的每个转换 $(s_j, a_j, r_{j+1}, s_{j+1})$，计算目标值 $y_j$：
     - 如果 $s_{j+1}$ 是终止状态，目标值就是奖励：$y_j = r_{j+1}$。
     - 否则，目标值使用**目标网络** $\hat{Q}$ 计算：
       
       $$
       y_j = r_{j+1} + \gamma \max_{a'} \hat{Q}(s_{j+1}, a'; \theta^-)
       $$
       
       $\gamma$ 是折扣因子。请注意，最大化操作是对目标网络对*下一*状态 $s_{j+1}$ 的输出进行的。
   - **梯度下降 (gradient descent)更新：** 对主 Q 网络权重 $\theta$ 执行梯度下降步骤。损失函数 (loss function)通常是预测 Q 值（来自主网络，针对已*采取*的动作 $a_j$）与计算出的目标值 $y_j$ 之间的均方误差 (MSE)：
     
     $$
     L(\theta) = \frac{1}{N} \sum_{j=1}^{N} (y_j - Q(s_j, a_j; \theta))^2
     $$
     
     使用 Adam 或 RMSprop 等优化器更新 $\theta$：$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$。
   - **更新目标网络：** 每隔 $C$ 步（其中 $C$ 是一个超参数 (parameter) (hyperparameter)），通过复制主网络权重来更新目标网络权重：$\theta^- \leftarrow \theta$。
   - **状态推进：** 设置 $s_t \leftarrow s_{t+1}$。
   - **检查终止：** 如果 $s_t$ 是终止状态，则结束当前幕。

> DQN 算法中的高级交互流程。智能体使用其 Q 网络进行动作选择，将经验存储在经验回放缓冲区中，采样批次以使用目标网络计算目标值，并通过梯度下降更新 Q 网络。目标网络定期从 Q 网络更新。

这种结构化方法，集成了经验回放和固定 Q 目标，使得 DQN 能够成功训练深度神经网络 (neural network)进行 Q-学习，让智能体可以直接从像素等高维感官输入中学习复杂行为，这是强化学习 (reinforcement learning)中的一项重要进展。下一节将介绍 DQN 中使用的神经网络的一些架构选择。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇原始论文介绍了深度Q网络（DQN）算法，展示了深度神经网络如何通过经验回放和目标网络从原始像素输入学习玩Atari游戏。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本标准的教科书，对强化学习做了全面的介绍，包括对DQN基础Q学习的清晰解释。(第二版)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — Joshua Achiam (2018)
  Publisher: OpenAI
  OpenAI提供的实用在线资源，对DQN等深度强化学习算法及其实现提供了清晰的解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本书介绍了深度学习的基础知识，为理解DQN等算法中使用的神经网络组件提供了必要的背景。
