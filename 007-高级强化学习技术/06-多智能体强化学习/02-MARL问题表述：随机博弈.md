# MARL问题表述：随机博弈

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/marl-stochastic-games)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为有效地设计多个交互智能体的算法，需要一个能扩展单智能体强化学习 (reinforcement learning)中常用的马尔可夫决策过程（MDP）模型的框架。不确定环境下多智能体序列决策的标准数学模型是**随机博弈**，通常也称为**马尔可夫博弈**。这种形式化为理解和应对智能体交互带来的复杂性奠定了基础。

随机博弈（SG）通过引入多个智能体来泛化MDP，这些智能体的行动共同影响状态转移和奖励。让我们形式化定义其组成部分：

1. **智能体有限集合**， $\mathcal{N} = \{1, 2, ..., N\}$，其中 $N \ge 2$。
2. **状态空间**， $\mathcal{S}$，表示环境的可能配置。通常假定所有智能体共享此空间或对其完全可观测，尽管针对更复杂的情况存在部分可观测的扩展（POSG 或 Dec-POMDPs）。
3. **每个智能体的行动空间**， $\mathcal{A}_i$。每个智能体 $i$ 选择一个行动 $a_i \in \mathcal{A}_i$。
4. **联合行动空间**， $\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times ... \times \mathcal{A}_N$。联合行动 $\mathbf{a} = (a_1, a_2, ..., a_N) \in \mathcal{A}$ 由每个智能体选择的一个行动组成。
5. **状态转移概率函数**，$P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0, 1]$。此函数定义了在给定当前状态 $s \in \mathcal{S}$ 和**联合行动** $\mathbf{a} \in \mathcal{A}$ 的情况下，转移到下一个状态 $s' \in \mathcal{S}$ 的概率。我们将其写为 $P(s' | s, \mathbf{a})$。请注意与 MDP 的重要区别：下一个状态的分布取决于*所有*智能体而非仅一个智能体的行动。
6. **每个智能体的奖励函数**，$R_i: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$。每个智能体 $i$ 根据当前状态 $s$、**联合行动** $\mathbf{a}$ 和可能的下一个状态 $s'$ 接收奖励 $r_i$。奖励 $r_i = R_i(s, \mathbf{a}, s')$ 特定于智能体 $i$，并且通常取决于每个智能体的行动。
7. **折扣因子**， $\gamma \in [0, 1)$，所有智能体共享，用于计算累积奖励。

正如在MDP中一样，智能体根据策略选择行动。在MARL中，我们处理一个**联合策略** $\boldsymbol{\pi} = (\pi_1, \pi_2, ..., \pi_N)$，其中每个 $\pi_i$ 是智能体 $i$ 的策略。智能体 $i$ 的策略， $\pi_i: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_i)$，将状态映射到其行动 $\mathcal{A}_i$ 的概率分布。如果策略是确定性的，则 $\pi_i: \mathcal{S} \rightarrow \mathcal{A}_i$。

### 交互的可视化

想象一个简单情境，两个机器人（智能体 1 和智能体 2）需要协调将一个箱子推到目标位置（状态）。

> 图示说明了当前状态 $s$ 如何被两个智能体观察。它们独立选择行动 $a_1$ 和 $a_2$，形成联合行动 $\mathbf{a}$。这个联合行动连同状态 $s$ 决定了转移到下一个状态 $s'$ 的概率，并确定了每个智能体的个体奖励 $r_1$ 和 $r_2$。

### 非平稳性正式说明

随机博弈的表述明确指出了章节引言中提及的**非平稳性**问题。考虑智能体 $i$ 的视角。它根据其策略 $\pi_i(a_i | s)$ 选择行动 $a_i$。环境的响应（下一个状态 $s'$ 和奖励 $r_i$）取决于*联合*行动 $\mathbf{a} = (a_i, \mathbf{a}_{-i})$，其中 $\mathbf{a}_{-i}$ 表示所有其他智能体的行动。

如果其他智能体的策略 $\boldsymbol{\pi}_{-i}$ 正在改变（正如它们在学习过程中通常所发生的那样），那么*从智能体 $i$ 的视角来看*，有效转移动态 $P(s' | s, a_i)$ 和奖励函数 $R_i(s, a_i)$ 也会改变，即使底层博弈动态 $P(s' | s, \mathbf{a})$ 和 $R_i(s, \mathbf{a}, s')$ 是固定的。这违反了标准单智能体强化学习 (reinforcement learning)算法（如Q学习）所要求的马尔可夫性质假设，使得直接应用存在问题。环境显得非平稳，因为其他学习智能体的行为是智能体 $i$ 有效环境的一部分。

### 多智能体交互类型

随机博弈可以建模各种交互类型，其特征在于智能体的奖励结构：

- **完全合作：** 所有智能体共享完全相同的奖励函数：$R_1 = R_2 = ... = R_N$。它们的目标是共同成功。示例包括机器人团队的协作或同步任务。
- **完全竞争（零和）：** 智能体之间目标完全对立。对于两个智能体，$R_1 = -R_2$。通常，对于所有 $s, \mathbf{a}, s'$，$\sum_{i=1}^N R_i(s, \mathbf{a}, s') = 0$。像国际象棋或围棋（不考虑平局）这样的经典棋类游戏属于此类。
- **混合（一般和）：** 这是最普遍的情况，包含兼具合作和竞争元素的情境。智能体拥有不一定对齐 (alignment)或直接对立的个体奖励函数。示例包括交通导航、资源共享或经济市场。

### 目标和解决方案思想

在单智能体强化学习 (reinforcement learning)中，目标通常是找到一个策略 $\pi$，以最大化预期折扣累积奖励，由价值函数 $V^{\pi}(s)$ 或 Q 函数 $Q^{\pi}(s, a)$ 表示。

在MARL中，目标取决于博弈类型。

- 在**合作**环境中，目标通常是找到一个联合策略 $\boldsymbol{\pi}$，以最大化共同目标，例如共享的预期回报 $\mathbb{E}[\sum_{t=0}^\infty \gamma^t R(s_t, \mathbf{a}_t, s_{t+1}) | s_0=s, \boldsymbol{\pi}]$。
- 在**竞争或混合**环境中，情况更为复杂。智能体可能旨在最大化自己的奖励，这可能导致博弈论中的一些思想，例如寻找**纳什均衡**。纳什均衡是一个联合策略 $\boldsymbol{\pi}^* = (\pi_1^*, ..., \pi_N^*)$，其中在假设所有其他智能体 $j \neq i$ 保持其策略 $\pi_j^*$ 不变的情况下，没有单个智能体 $i$ 可以通过单方面改变其策略 $\pi_i^*$ 来提高其预期回报。

我们可以根据联合策略 $\boldsymbol{\pi}$ 定义智能体特定的价值函数：

- 智能体 $i$ 的状态价值函数：$V_i^{\boldsymbol{\pi}}(s) = \mathbb{E}_{\boldsymbol{\pi}}[\sum_{t=0}^\infty \gamma^t R_i(s_t, \mathbf{a}_t, s_{t+1}) | s_0=s]$
- 智能体 $i$ 的行动价值函数：$Q_i^{\boldsymbol{\pi}}(s, \mathbf{a}) = \mathbb{E}_{\boldsymbol{\pi}}[\sum_{t=0}^\infty \gamma^t R_i(s_t, \mathbf{a}_t, s_{t+1}) | s_0=s, \mathbf{a}_0=\mathbf{a}]$

理解随机博弈框架非常重要，因为它精确定义了MARL算法旨在解决的问题。它强调了智能体之间的依赖关系，并提供了分析智能体交互所需的数学结构以及后续章节将处理的由此产生的非平稳性挑战。

## 参考资料

- [Stochastic Games](https://pnas.org/doi/10.1073/pnas.39.10.1095) — Lloyd S. Shapley (1953)
  Journal: Proceedings of the National Academy of Sciences of the United States of America; Publisher: National Academy of Sciences; Volume: 39; Pages: 1095-1100; DOI: [10.1073/pnas.39.10.1095](https://doi.org/10.1073/pnas.39.10.1095)
  首次正式定义了随机博弈，为多智能体序列决策建立了理论框架。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  提供了强化学习的通用介绍（第二版），包括对博弈论和多智能体方面的讨论，为多智能体强化学习提供了基础背景。
- [Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations](https://www.cambridge.org/9780521899437) — Yoav Shoham, Kevin Leyton-Brown (2009)
  Publisher: Cambridge University Press
  这部学术著作深入探讨了多智能体系统基础、详细的博弈论、重复博弈和随机博弈。
- [Multi-Agent Reinforcement Learning: An Overview of Concepts and Applications](https://link.springer.com/book/10.1007/978-3-642-13627-1) — Lucian Buşoniu, Robert De Schutter, and Robert Babuška (2010)
  Publisher: Springer; DOI: [10.1007/978-3-642-13627-1](https://doi.org/10.1007/978-3-642-13627-1)
  一本专门的教科书，概述了多智能体强化学习，重点在于随机博弈的公式化和相关解决方案。
- [Multi-Agent Reinforcement Learning: A Comprehensive Survey](https://www.nowpublishers.com/article/Details/MAL-083) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu and Sen Zhao (2021)
  Journal: Foundations and Trends® in Machine Learning; Publisher: now publishers; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  这项广泛的综述提供了多智能体强化学习的最新视角，涵盖了随机博弈、非平稳性以及当前的解决方案。

---

[上一节](01-%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E4%BB%8B%E7%BB%8D.md) · [下一节](03-%E9%9B%86%E4%B8%AD%E5%BC%8F%E4%B8%8E%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A7%E5%88%B6.md)
