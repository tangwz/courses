# 集中式训练与去中心化执行 (CTDE)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/ctde-marl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在多智能体环境中独立训练智能体（例如使用独立Q学习或IQL）面临一大挑战：非平稳性。从智能体$i$的角度看，随着其他智能体（$j \neq i$）更新它们的策略，环境似乎在发生变化，使得智能体$i$的学习过程难以收敛。这就像试图击中一个会移动并对你的尝试做出反应的目标。

另一方面，一个完全中心化的控制器虽然能观察一切并为所有智能体指定动作，从而避免非平稳性，但它的扩展性差，并且在执行时通常需要不切实际的通信能力。想象一下，你需要一个中央指挥中心来实时协调多个机器人的每一步动作。这通常不切实际。

### CTDE方法

集中式训练与去中心化执行 (CTDE) 提供了一种有吸引力的折衷方案。它的目标是同时实现两种优势：在学习阶段运用全局信息来稳定训练并减轻非平稳性，同时仍能生成允许智能体在执行时仅使用其局部观察进行独立行动的策略。

主要思想简单而有效：**训练时使用额外信息，执行时使用局部信息。**

> CTDE设置中的信息流。在训练期间（蓝色背景），一个集中式组件使用全局信息来指导去中心化行动者（浅蓝色背景）的学习。在执行期间，智能体仅使用其局部观察 $o_i$ 来选择动作 $a_i$。

### 工作原理

1. **集中式训练：** 在训练阶段，我们假定可以获取比执行时更多信息。这可能包括*所有*智能体的观察和动作，甚至环境的底层全局状态 $s$。这些额外信息通常输入到一个集中式组件中，在行动者-评论器架构中通常是一个评论器。

   - 此评论器可以学习一个联合动作价值函数 $Q_{tot}(s, \mathbf{a})$，其中 $\mathbf{a} = (a_1, ..., a_N)$ 是所有智能体的联合动作；或者它可能学习基于额外全局信息调整的单个智能体价值函数 $Q_i(s, a_i)$。
   - 通过获取其他智能体的动作和观察（或全局状态），集中式评论器能获得更稳定的学习信号。它能更好地评估智能体 $i$ 的动作 $a_i$ 在*所有*其他智能体行为背景下的价值，从而有效地稳定由非平稳性引起的“移动目标”问题。
   - 集中式组件为更新单个智能体的策略（行动者）提供指引（例如，梯度）。
2. **去中心化执行：** 训练完成后，集中式组件被移除。每个智能体 $i$ 部署其学得的策略 $\pi_i(a_i | o_i)$，该策略*仅*基于自身的局部观察历史 $o_i$ 来选择动作。

   - 策略本身仅基于局部信息调整，使得执行完全去中心化。智能体在运行时无需通信或观察其他智能体的状态/动作。

### CTDE的优点

- **解决非平稳性：** 通过在训练期间为学习算法提供更全面的视角，CTDE 相较于独立学习器能显著稳定学习。
  "\* **实用的执行：** 生成去中心化策略，适用于在执行时通信或全局观察可能受限或成本高昂的场景。"
- **灵活性：** CTDE 框架通用，支撑着许多为不同多智能体设置（合作、竞争、混合）设计的成功MARL算法。我们将在本章后面遇到使用此方法的具体例子，如MADDPG和VDN/QMIX。

### 考量因素

尽管功能强大，但CTDE并非万能药。主要考虑点是确保使用集中信息学习到的去中心化策略，在执行时移除这些额外信息后，仍能表现良好。这通常需要仔细设计集中式训练机制和各个策略。此外，虽然*执行*是去中心化的，但*训练*过程仍需要一种收集和处理可能大量全局信息的机制，这可能会根据问题规模和具体算法带来其自身的计算瓶颈。

总而言之，CTDE是MARL中一种非常有效且被广泛采用的方法。它在训练期间直接处理非平稳性问题，同时保持去中心化执行的实用性，为许多先进的多智能体算法奠定了基础。

## 参考资料

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper_files/paper/2017/file/68a9750337a418a86fe06c1991a0ad99-Paper.pdf) — Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 6379-6390; DOI: [10.5591/978-1-57783-000-8-124](https://doi.org/10.5591/978-1-57783-000-8-124)
  这篇基础性论文介绍了多智能体深度确定性策略梯度（MADDPG），这是一种广泛使用的算法，它在混合合作-竞争环境中展示了CTDE方法。
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) — Tabish Rashid, Gregory Farquhar, Shimon Whiteson, Michael Wooldridge (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); DOI: [10.5555/3326779.3326815](https://doi.org/10.5555/3326779.3326815)
  这篇论文提出了QMIX，这是一种用于合作MARL的著名CTDE算法，它使用一个中心化混合网络来分解联合动作值函数，同时确保去中心化执行。
- [Value-Decomposition Networks For Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1706.05296) — Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, Thore Graepel (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.05296](https://doi.org/10.48550/arXiv.1706.05296)
  这项工作介绍了价值分解网络（VDN），这是一种有影响力的早期CTDE方法，它将总Q值分解为个体智能体Q值，用于合作多智能体任务，为QMIX等后续方法奠定了基础。

---

[上一节](06-%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB%E6%96%B9%E6%B3%95.md) · [下一节](08-%E4%BB%B7%E5%80%BC%E5%88%86%E8%A7%A3%E6%96%B9%E6%B3%95%20%28VDN%2C%20QMIX%29.md)
