---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "offline-vs-online-offpolicy"
sourceId: 3484
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/offline-vs-online-offpolicy"
title: "离线强化学习与在线及离策略强化学习的区别"
description: "对比离线强化学习与标准在线及离策略学习设置。"
order: 2
plots: []
sourceHash: "f1b9050d219b44060cfc2cecc14c08407e4e94879feed705f4e330bb66d75f61"
sourceCorrections: []
---

我们来阐明离线强化学习 (reinforcement learning)与我们之前遇到的在线及标准离策略方法有何不同。理解这些区别对于了解仅从数据日志学习所涉及的特殊难题和技术至关重要。

在**在线强化学习 (Online RL)** 中，智能体是积极参与者。它持续与环境互动：在状态 $s_t$ 中采取动作 $a_t$，观察下一个状态 $s_{t+1}$ 和奖励 $r_t$，并利用这份新经验 $(s_t, a_t, r_t, s_{t+1})$ 实时更新其策略 $\pi$ 或价值函数 $Q$。可以想象 SARSA 或基础的 Actor-Critic 算法在环境循环中一步步操作。智能体自主掌控数据收集过程，平衡尝试新行为和运用已有知识。如果需要某个状态-动作区域的更多信息，它便可以前往该处进行试验。

**标准离策略强化学习 (Standard Off-Policy RL)**，由 DQN 或 DDPG 等算法实现，具有一个特点。智能体学习一个目标策略 $\pi$（通常根据当前价值估计采取贪婪策略），同时可能按照一个不同的策略 $\pi_b$（例如， $\pi$ 的 $\epsilon$-贪婪版本）行动。它将通过 $\pi_b$ 收集的经验存储在回放缓冲区 $\mathcal{D}$ 中。之后通过从该缓冲区抽取小批量数据来进行更新。虽然更新使用了由不同策略生成的潜在“旧”数据（因此是“离策略”），但要点是智能体*仍在与环境互动*。它不断向缓冲区 $\mathcal{D}$ 添加其当前行为策略 $\pi_b$ 收集的新转换。这种持续涌入的新数据，在不断变化的行为策略指引下，使智能体最终获取关于其目标策略相关状态-动作对的信息，即使它们最初未被充分试验。这有助于减轻（但不能消除）因评估行为策略通常不会采取的动作而产生的问题。

**离线强化学习 (Offline RL)（批量强化学习 Batch RL）** 则是一种更为受限的环境。在这里，智能体不再与环境互动。它获得一个固定、静态的数据集 $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$，该数据集通常由之前使用某些未知或部分已知的行为策略 $\pi_b$（或多个策略）收集。学习算法必须仅利用这批数据来构建尽可能好的策略 $\pi$。

> 在线、离策略（在线）和离线强化学习设置中的数据流对比。在线强化学习涉及直接互动。离策略（在线）使用回放缓冲区但仍会与环境互动以添加新数据。离线强化学习仅从预先设定的固定数据集学习。

这种“无互动”的限制是离线强化学习的显著特点和主要难点。主要区别在于：

1. **数据覆盖范围固定：** 智能体无法通过采取在数据集 $\mathcal{D}$ 中未充分体现的动作来提问“如果那样做会怎样？”。如果行为策略 $\pi_b$ 从未涉足状态-动作空间中某些可能有利的区域，智能体就没有直接方法来了解它们。学习策略 $\pi$ 的质量受到 $\mathcal{D}$ 的质量和覆盖范围的根本性制约。
2. **分布偏差加剧：** 这是前面提到的主要难题。标准离策略方法在目标策略 $\pi$ 与行为策略 $\pi_b$ 显著不同时，已经可能出现错误。这是因为价值估计常依赖于自举（例如，在 Q-学习中使用 $\max_{a'} Q(s', a')$）。如果与目标策略相关的状态-动作对 $(s', a')$ 相对于数据集 $\mathcal{D}$ 属于分布外 (OOD)，则 $Q(s', a')$ 的估计值可能因数据不足而非常不准确。若无环境互动来纠正这些错误或收集相关数据，这些不准确性在训练中会不断累加，导致策略表现不佳或偏离。在线离策略方法有时能够通过最终在这些 OOD 区域收集数据来弥补，但离线方法无法做到。
3. **反事实查询存在困难：** 评估或优化一个策略 $\pi$ 需要估计策略 $\pi$ 会采取的动作的价值，这可能与数据集 $\mathcal{D}$ 中给定状态 $s_i$ 处实际存在的动作 $a_i$ 有显著差异。标准监督学习 (supervised learning)依赖于训练和测试数据均取自同一分布。离线强化学习在 $\pi \neq \pi_b$ 时，本质上就违反了这一假设。

以下表格总结了这些对比：

| 特点 | 在线强化学习 | 离策略强化学习 (在线) | 离线强化学习 (批量强化学习) |
| --- | --- | --- | --- |
| **数据来源** | 主动互动 | 主动互动 + 回放缓冲区 | 固定、预收集数据集 |
| **互动** | 持续 | 持续 | 学习期间无 |
| **探索** | 智能体积极探索 | 智能体积极探索 (通过 $\pi_b$) | 受限于固定数据集中的数据 |
| **学习策略** | 通常为同策略 ($\pi = \pi_b$) | 离策略 ($\pi \neq \pi_b$) | 离策略 ($\pi \neq \pi_b$) |
| **主要难题** | 探索-利用权衡 | 样本效率、离策略稳定性 | **分布偏差**、数据覆盖 |
| **错误修正** | 通过新的环境互动 | 通过新的环境互动 | 无互动修正；依赖算法设计 |

因此，为离线强化学习设计的算法必须明确纳入考量缺乏互动以及分布偏差的潜在严重影响。它们通常结合机制，要么限制所学策略与行为策略的分布保持“接近”，要么对价值估计进行规范，使其对分布外动作持保守态度。我们将在后续章节中介绍这些专门技术。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  提供了强化学习的基础知识，包括在线和离策略方法的概念。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis (2015)
  Journal: Nature; Publisher: Springer Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了深度Q网络（DQN），一种重要的离策略算法，展示了通过与环境交互从回放缓冲区学习的方法。
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643) — Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu (2020)
  DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643)
  全面概述了离线强化学习，详细阐述了其定义、挑战以及应对分布偏移等问题的方法。
