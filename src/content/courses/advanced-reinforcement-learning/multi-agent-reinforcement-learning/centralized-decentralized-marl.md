---
course: "advanced-reinforcement-learning"
chapter: "multi-agent-reinforcement-learning"
lesson: "centralized-decentralized-marl"
sourceId: 3468
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/centralized-decentralized-marl"
title: "集中式与分布式控制"
description: "比较完全集中式训练/执行与分布式方法。"
order: 3
plots: []
sourceHash: "ccef31c73b22bc4041ab647e86490aba1efbfc079a601e8d98bce701ce18cee4"
sourceCorrections: []
---

在设计包含彼此影响的智能体的系统时，一个基本考量是如何分配控制和学习。每个智能体在训练时能获取多少信息？在执行时，它使用多少信息来做决策？这些问题引出了各种方法，从完全集中式控制到完全分布式控制不等，实践中，折中方案通常最有效。了解这些模式对于选择和应用合适的多智能体强化学习 (reinforcement learning)算法非常重要。

### 完全集中式控制

设想有一个单一、无所不知的控制器管理系统中的每个智能体。这就是完全集中式控制的要义。在这种方法中：

- **训练：** 一个单一的学习算法或策略函数接收*全局状态*（或所有智能体观测的串联）并同时为所有智能体输出一个*联合动作*。奖励信号，通常是全局团队奖励，用于更新这个中央策略。从这个中央学习者的角度来看，多智能体问题本质上转化为一个单智能体问题，尽管其状态空间和动作空间可能非常庞大。
- **执行：** 中央控制器在每一步观察全局状态，并指定每个智能体的动作。

> 中央控制器根据全局状态为所有智能体做决策。

**优点：**

- **最佳协作：** 理论上，中央控制器可以学习全局最佳的协作策略，因为它能获取所有必要信息。
- **平稳性：** 学习过程不会因为其他智能体适应而导致非平稳性，因为只有一个学习实体控制所有事物。

**缺点：**

- **扩展性问题：** 主要缺点是维度灾难。联合状态空间 $S = S_1 \times S_2 \times \dots \times S_N$，特别是联合动作空间 $A = A_1 \times A_2 \times \dots \times A_N$，会随着智能体数量 $N$ 的增加呈指数级增长。这使得即使是中等规模的多智能体系统，学习也变得难以处理。
  "\* **完全可观察性要求：** 假设中央控制器可以获取完整的全局状态，这在实践应用中通常不切实际，因为存在部分可观察性或通信限制。"
- **集中式执行：** 在执行过程中需要一个中央实体，这可能成为瓶颈、单点故障，并且在要求分布式操作的场景（例如独立自主的机器人）中不实用。

由于这些限制，完全集中式控制对于复杂的多智能体强化学习 (reinforcement learning)问题通常不可行。

### 完全分布式控制（独立学习者）

在另一个极端，是完全分布式方法，通常通过*独立学习者*实现。具体来说：

- **训练：** 每个智能体 $i$ 仅根据其*局部观测* $o_i$ 和*个体奖励* $r_i$（有时也可能是共享的全局奖励）学习自己的策略 $\pi_i$。每个智能体都将其他智能体视为环境动态的一部分。标准的单智能体强化学习 (reinforcement learning)算法（如Q-learning、DQN、PPO）可以独立地应用于每个智能体。
- **执行：** 每个智能体仅根据其局部观测，使用其学到的策略 $\pi_i(a_i | o_i)$ 来选择动作。

> 独立学习者仅使用局部观测和奖励进行操作，并将其他智能体视为环境的一部分。

**优点：**

- **扩展性：** 每个智能体的学习复杂度不直接随智能体总数增加而扩展。更容易应用于大型系统。
- **简单性：** 通过重用现有单智能体强化学习代码库，实现起来相对直接。
- **分布式执行：** 天然支持分布式执行，运行时无需中央协调器。

**缺点：**

- **非平稳性：** 这是主要障碍。从智能体 $i$ 的角度来看，环境表现出非平稳性，因为其他智能体 $j \neq i$ 正在同时学习并改变其策略 $\pi_j$。这违反了大多数单智能体强化学习算法所基于的平稳性假设，可能导致学习不稳定或效率低下。
- **次优协作：** 智能体根据局部信息进行贪婪学习，通常无法收敛到协作或全局最佳策略。它们可能学习到冲突的行为或发生震荡。
- **信用分配：** 如果使用全局奖励，个体智能体很难仅根据局部观测来判断其对团队成功或失败的具体贡献。

尽管其简单，非平稳性问题常常阻碍完全分布式独立学习者的表现，尤其是在需要高度协作的任务中。

### 集中式训练与分布式执行 (CTDE)

认识到完全集中式和完全分布式极端方法的局限性，*集中式训练与分布式执行* (CTDE) 方法已成为一种非常有效且受欢迎的折中方案。

- **训练：** 在学习阶段（通常在模拟中），算法会借助*集中式*信息。这可能包括全局状态、所有智能体的观测，甚至所有智能体采取的动作。这些额外信息有助于稳定训练，解决非平稳性问题，并学习协作行为。例如，一个集中式评论家可以根据全局信息评估联合动作，而分布式行动者则学习各自的策略。
- **执行：** 重要的是，一旦训练完成，每个智能体仅使用其*局部*观测来执行其策略，就像完全分布式情况一样。训练期间使用的集中式组件在部署时会被丢弃。

> CTDE 在训练时使用集中式信息，但在执行时仅依赖局部信息。

**优点：**

- **平衡性能与实用性：** 在训练时借助全局信息克服非平稳性并改善协作，同时仍允许实用的分布式执行。
- **广泛适用性：** 是许多成功的多智能体强化学习 (reinforcement learning)算法的依据，包括 MADDPG、VDN 和 QMIX，我们将在本章后续内容中讨论这些算法。
- **提升稳定性：** 集中式组件可以大大稳定学习过程。

**缺点：**

- **训练复杂度：** 需要在训练期间有效地整合集中式信息，并确保其能转化为有效的分布式策略的机制。
- **模拟要求：** 通常依赖于在训练/模拟阶段能获取集中式信息，但这并非总能实现。
- **信用分配依然存在：** 尽管有所缓解，但在执行过程中根据局部观测进行信用分配的挑战依然存在，不过集中式训练提供了更多信息来处理这个问题。

### 选择正确的方法

集中式、分布式和 CTDE 方法之间的选择在很大程度上取决于多智能体问题的具体情况：

- **可观察性：** 智能体能否观察完整的全局状态，还是只有部分局部信息？
- **通信：** 智能体在执行时能否通信？是否存在可靠的中央协调器？
- **扩展性：** 涉及多少智能体？
- **任务性质：** 任务是需要紧密协作（偏向集中式信息），还是允许更独立的操作？
- **执行限制：** 分布式执行是否是严格要求？

实践中，由于完全集中式控制的扩展性限制以及完全分布式学习的非平稳性问题，**CTDE 为许多多智能体强化学习 (reinforcement learning)任务提供了一个引人关注且广泛采用的框架。** 它代表了一种务实的方式，可在不牺牲分布式执行必要性的前提下，融入集中式信息的优势。后续章节将介绍基于这些模式的特定算法，尤其侧重于 CTDE 方法。

## 参考资料

- [A Survey of Multi-Agent Reinforcement Learning Research](https://ieeexplore.ieee.org/document/9583162) — Kaiqing Zhang, Zhuoran Yang, Tamer Basar (2021)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 109; Pages: 1720-1740; DOI: [10.1109/JPROC.2021.3117521](https://doi.org/10.1109/JPROC.2021.3117521)
  对多智能体强化学习的广泛概述，涵盖了其挑战（如非平稳性）和各种控制范式。
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.02275](https://doi.org/10.48550/arXiv.1706.02275)
  介绍了MADDPG，这是一种CTDE算法，可解决多智能体环境中的非平稳性问题。
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) — Tabish Rashid, Gregory Farquhar, Shimon Whiteson, Jakob Foerster (2018)
  Journal: International Conference on Machine Learning (ICML); Pages: 4295-4304; DOI: [10.48550/arXiv.1803.01148](https://doi.org/10.48550/arXiv.1803.01148)
  提出了QMIX，一种广泛用于合作型多智能体强化学习的CTDE方法，强调了价值函数分解。
- [Multi-Agent Reinforcement Learning: Theory and Algorithms](https://link.springer.com/book/10.1007/978-981-15-2831-2) — Wen-Qiang Zhang, Xin Liu, Ya-Li Du (2020)
  Publisher: Springer Nature Singapore; DOI: [10.1007/978-981-15-2831-2](https://doi.org/10.1007/978-981-15-2831-2)
  一本关于多智能体强化学习理论的书籍，包含了集中式、分布式和CTDE方法。
