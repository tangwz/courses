---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "offline-rl-introduction"
sourceId: 3483
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/offline-rl-introduction"
title: "离线强化学习（批处理强化学习）简介"
description: "离线强化学习的缘由与定义：从预先收集的数据中进行学习。"
order: 1
plots: []
sourceHash: "acdfe7baf436b56e5be570bd6aa8dabbbd3b6d2e76a187252428bd56efad5aee"
sourceCorrections: []
---

前几章详细介绍了智能体通过与环境持续互动、收集数据并循环更新策略的学习方法。现在，我们将注意力转向一种不同但日益重要的学习方式：**离线强化学习 (reinforcement learning)**，通常也称为**批处理强化学习**。

### 为何选择离线学习？

设想一下，在某些情况下，将进行试探的强化学习 (reinforcement learning)智能体直接部署到真实系统中是不切实际、不安全或成本极高的。例如：

- **医疗保健：** 直接在患者身上测试新的治疗策略是不道德的。从现有（匿名化）患者记录中学习是更可行的方法。
- **机器人：** 允许物理机器人在初始学习阶段随机试探可能导致昂贵的损坏或不安全的情况。从记录的传感器数据或演示中学习是更优的选择。
- **自动驾驶：** 在公共道路上进行试探具有显著风险。使用人类驾驶员或现有系统收集的大量驾驶数据日志提供更安全的替代方案。
- **推荐系统/在线广告：** 公司通常拥有海量的过往用户交互日志。从这些历史数据中学习改进的推荐或竞价策略，同时又不通过可能次优的试探性动作来干扰实时系统，这是非常可取的。

在这些及许多其他情况下，我们面对的是预先收集好的固定交互数据集，这些数据可能由不同策略甚至多个策略生成。目标是借助这一静态数据集，在学习过程中*不*与环境进行任何进一步交互的情况下，学习到尽可能好的策略。这是离线强化学习的主要思想。

### 离线强化学习 (reinforcement learning)问题设置

离线强化学习是一种强化学习方法，其中智能体通过一个静态数据集 $D$ 来学习策略，该数据集由一组转换组成。


$$
D = \{ (s_i, a_i, r_i, s'_i) \}_{i=1}^N
$$


这里，$s_i$ 是状态，$a_i$ 是在该状态下采取的动作，$r_i$ 是获得的奖励，而 $s'_i$ 是结果的下一个状态。这个数据集 $D$ 是使用一个或多个*行为策略*收集的，统称为 $\pi_b$。重要的是，我们可能不知道生成数据的确切策略 $\pi_b$。主要限制是我们的学习算法*只能*使用 $D$ 中存在的转换。它不能向环境查询新的状态-动作对的结果。

目标是使用这个数据集 $D$ 来学习一个目标策略 $\pi$，使其在实际环境中部署时能够最大化预期累积奖励。

> 离线强化学习流程：数据通过环境交互并采用行为策略进行收集，形成一个静态数据集。离线强化学习算法仅从该数据集学习新策略，无需进一步的环境交互。

### 与在线和标准离策略学习的对比

需要区分开离线强化学习 (reinforcement learning)与相关内容：

1. **在线强化学习：** 智能体持续与环境互动，收集经验（转换），并根据流入的数据流更新其策略。试探是学习循环中活跃组成部分。示例包括SARSA、A2C/A3C（在线运行时）和在线Q学习。
2. **离策略强化学习（在线设置）：** 智能体在遵循不同的行为策略 $\pi_b$（例如，epsilon-greedy）收集数据时，学习目标策略 $\pi$（例如，贪婪策略）。DQN和DDPG等算法是离策略的。然而，在标准*在线*离策略设置中，智能体*仍与环境互动*，采用 $\pi_b$ 收集新的转换并将其添加到其回放缓冲区。数据集会随时间增长和演变。
3. **离线强化学习（批处理强化学习）：** 这是离策略学习的一种更严格形式。与标准离策略方法类似，它从由 $\pi_b$ 生成的数据中学习策略 $\pi$。然而，其主要特点是数据集 $D$ 是*固定的*，并且完全在学习开始*之前*收集。在训练阶段，绝对不会与环境进行进一步互动。

### 分布偏移的隐患

为什么纯粹从静态数据集学习如此困难？主要障碍是**分布偏移**。固定数据集 $D$ 反映了行为策略 $\pi_b$ 引起的状态-动作访问频率。如果我们的学习算法试图评估或改进一个行为与 $\pi_b$ 不同的目标策略 $\pi$，那么 $\pi$ 可能会偏好在 $D$ 中表示不佳或完全不存在的动作或状态。

思考一下，在离线数据集 $D$ 上采用标准离策略算法（如DQN）训练Q网络。贝尔曼更新涉及诸如 $\max_{a'} Q(s', a')$ 之类的项。如果在数据集中遇到状态 $s'$，但动作 $a' = \arg\max_{a'} Q(s', a')$ 在状态 $s'$ 中*从未*被 $\pi_b$ 采取过，那么Q值 $Q(s', a')$ 就纯粹通过函数逼近外推来估计。神经网络 (neural network)在训练数据分布之外进行外推时是出了名的不可靠。这可能导致极其不准确的Q值估计，通常是显著的高估，从而导致学习过程发散或收敛到次优策略。这种失败模式，源于查询批次中不存在的特定状态下动作的价值，是分布偏移的直接结果。

应对这一挑战是现代离线强化学习 (reinforcement learning)研究的核心议题。接下来的章节将探讨专门为处理分布偏移而开发的技术，包括离线策略评估的方法以及为稳定的离线策略优化而设计的算法。

## 参考资料

- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643) — Sergey Levine, Aviral Kumar, George Tucker, Justin Fu (2020)
  Journal: Foundations and Trends in Machine Learning; Volume: 13; Pages: 1-174; DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643)
  全面概述离线强化学习，包括其动机、问题设置、挑战和各种算法方法。
- [Batch Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-3-642-27645-3_4) — Sascha Lange, Martin Riedmiller, Alois Knoll (2012)
  Publisher: Springer Berlin Heidelberg; Pages: 83-108; DOI: [10.1007/978-3-642-27645-3_4](https://doi.org/10.1007/978-3-642-27645-3_4)
  提供早期批处理强化学习概念和技术的详细说明的图书章节。
- [Off-Policy Deep Reinforcement Learning without Exploration](http://proceedings.mlr.press/v97/fujimoto19a.html) — Scott Fujimoto, David Meger, Doina Precup (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 97; Pages: 2052-2062; DOI: [10.5555/3305890.3306013](https://doi.org/10.5555/3305890.3306013)
  介绍批处理约束Q学习（BCQ），这是一种旨在缓解离线深度强化学习中分布偏移问题的开创性算法。
- [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) — Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 1172-1183; DOI: [10.48550/arXiv.2006.04779](https://doi.org/10.48550/arXiv.2006.04779)
  提出保守Q学习（CQL），这是一种通过学习Q函数下限来解决离线RL中超出分布动作导致的过高估计误差的原则性方法。
