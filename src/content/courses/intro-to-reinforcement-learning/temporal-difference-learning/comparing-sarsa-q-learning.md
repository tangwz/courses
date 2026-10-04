---
course: "intro-to-reinforcement-learning"
chapter: "temporal-difference-learning"
lesson: "comparing-sarsa-q-learning"
sourceId: 1647
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/comparing-sarsa-q-learning"
title: "比较 SARSA 与 Q-学习"
description: "突出 SARSA 与 Q-学习在更新规则和行为方式上的差异。"
order: 6
plots: []
sourceHash: "ed4b88c2484e2d4c4c801995d2381d62c7b5d5203225ce9191b5d3397bf99b2e"
sourceCorrections: []
---

直接比较 SARSA 和 Q-学习两种算法是很有价值的。它们都是基础的时序差分 (TD) 控制方法，旨在通过估计动作值函数 ($Q$ 值) 来学习最优行为。它们共享 TD 自举的核心思想——根据其他已学估计值更新估计值。然而，它们的更新规则和学习原理存在显著差异，主要源于 SARSA 是一种*在线策略*算法，而 Q-学习是一种*离线策略*算法。

### 更新规则差异

核心区别在于每种算法如何计算更新中使用的目标值。让我们回顾一下在采取动作 $A_t$ 并获得奖励 $R_{t+1}$ 后，从状态 $S_t$ 到 $S_{t+1}$ 转移的更新规则。

**SARSA 更新：**
SARSA 代表 State-Action-Reward-State-Action。这个名称反映了其更新规则，该规则使用在状态 $S_{t+1}$ 中根据当前策略 $\pi$ 采取的*实际*动作 $A_{t+1}$。


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ \underbrace{R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})}_{\text{TD 目标 (使用实际的下一个动作 } A_{t+1})} - Q(S_t, A_t) ]
$$


目标值 $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ 取决于下一个状态 $S_{t+1}$ *和*由当前策略选择的下一个动作 $A_{t+1}$ (在学习过程中通常是 $\epsilon$-贪心策略)。SARSA 学习在状态 $S_t$ 中采取动作 $A_t$ 然后*此后遵循当前策略*的价值。

**Q-学习更新：**
另一方面，Q-学习使用下一个状态 $S_{t+1}$ 中*最大*可能的 Q 值来形成其目标，无论实际接下来采取的动作是什么。


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ \underbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)}_{\text{TD 目标 (使用下一个动作的最大 Q 值)}} - Q(S_t, A_t) ]
$$


目标值 $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ 代表所获得的奖励加上从状态 $S_{t+1}$ 中*最佳*可能动作的折扣值，根据当前的 Q 值估计。Q-学习学习在状态 $S_t$ 中采取动作 $A_t$ 然后*此后最优地 (贪婪地) 动作*的价值，即使智能体实际的下一个动作 $A_{t+1}$ (例如通过 $\epsilon$-贪婪行为选择) 不同。

### 在线策略与离线策略学习

目标更新的这种差异直接关联到在线策略/离线策略的区别：

- **SARSA (在线策略)：** 学习智能体当前*遵循的*策略 $\pi$ 的值函数 $Q^\pi$。如果智能体使用 $\epsilon$-贪婪策略进行行为，SARSA 学习 $\epsilon$-贪婪行为的价值。行为策略 (用于选择动作，例如 $A_{t+1}$) 和目标策略 (被学习的策略) 相同。
- **Q-学习 (离线策略)：** 学习最优*贪婪*策略的值函数 $Q^*$，无论智能体当前遵循何种策略来对环境进行行为。行为策略 (例如 $\epsilon$-贪婪) 可以与目标策略 (总是由 $\max_a Q(S_{t+1}, a)$ 暗示的贪婪策略) 不同。

### 行为后果

学习行为策略 (SARSA) 的价值与最优策略 (Q-学习) 的价值之间的差异可能导致不同的学习行为，特别是在行为可能存在风险的环境中。

考虑一个类似强化学习 (reinforcement learning)文献中常用的“悬崖行走”问题的场景。智能体需要找到从起点到目标状态的最短路径，但网格中的一个区域 ('悬崖') 如果进入会产生很大的负奖励。

> 悬崖行走环境中的路径差异。SARSA 倾向于学习一条更安全、可能更长的远离悬崖的路径，因为它考虑了在边缘附近次优行为动作的负面后果。Q-学习学习最短路径，假设最终会采取最优动作，忽略学习过程中使用的行为策略中固有的风险。

因为 SARSA 在其更新中包含实际选择的动作 $A_{t+1}$ (这可能是一个随机行为动作)，所以它学习到靠近悬崖的状态-动作对的 Q 值较低。这鼓励了一种避开危险区域的策略。Q-学习通过使用 $\max$ 运算符，学习到最优路径就在悬崖边缘，假设智能体在重要时不会犯行为错误。它对未来的动作更乐观。

### 收敛性

- **Q-学习：** 在标准随机近似条件下收敛到最优动作值函数 $Q^*$，前提是所有状态-动作对都持续被访问。它的收敛性与所遵循的策略无关。
- **SARSA：** 收敛到所遵循的特定策略 $\pi$ (例如 $\epsilon$-贪婪) 的动作值函数 $Q^\pi$。如果 $\epsilon$ 逐渐减小到零，SARSA 也会收敛到 $Q^*$。

### 总结表

| 特性 | SARSA | Q-学习 |
| --- | --- | --- |
| **类型** | 在线策略 TD 控制 | 离线策略 TD 控制 |
| **更新规则** | 使用 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ | 使用 $(S_t, A_t, R_{t+1}, S_{t+1})$ 和 $\max Q$ |
| **目标值** | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ |
| **学习内容** | *行为*策略的价值 (包含行为) | *最优* (贪婪) 策略的价值 |
| **行为方式** | 更保守，避免有风险的行为 | 更具进取性，直接学习最优路径 |
| **收敛性** | 收敛到 $Q^\pi$ (对于策略 $\pi$) | 收敛到 $Q^*$ (最优值) |

SARSA 和 Q-学习之间的选择取决于具体的应用场景。如果你需要在特定行为策略下评估或保证性能，或者如果学习过程中的安全性是主要考量 (避免灾难性行为动作)，SARSA 可能更受欢迎。如果主要目的是尽可能高效地找出最优策略，并且学习阶段的性能是次要的，Q-学习通常是更直接的方法，并为许多先进的深度强化学习 (reinforcement learning)算法奠定基础。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapters 6 and 6.4
  一本权威教科书，详细介绍了时序差分学习、SARSA、Q-学习以及策略内和策略外方法的区别，并包含如悬崖行走等说明性示例。
- [Q-learning](https://link.springer.com/article/10.1007/BF00992698) — Christopher J. C. H. Watkins and Peter Dayan (1992)
  Journal: Machine Learning; Volume: 8; Pages: 279-292; DOI: [10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
  介绍Q-学习的原始学术论文，这是一种基础的策略外时序差分控制算法。
- [On-line Q-learning using connectionist systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGOayUAiF7SeDvFqRbU486XQa0Ls0MTqz0dC8PquYkRHMPm-oNgBCNTomji-b6dXoJ4obPn9AG4lTbyBQpHaokIzsLFQlyaZ42MA0No3RmTP2bk63R-mVVqAxtr5kp8qAEgyPN5Y1Xofu-UsUg05eEheIX3aHJhzRybx797WkfnJeVaqA2IXYe9XtUxm0jUr2wNG7AQAB7Q-gY=) — Gavin A. Rummery and Mahesan Niranjan (1994)
  Journal: CUED/F-INFENG/TR 166, Cambridge University Engineering Department, Technical Report; Publisher: Cambridge University Engineering Department
  这份技术报告介绍了后来被明确命名为SARSA的算法，这是一种策略内时序差分控制方法。
- [Reinforcement Learning (Course Lectures)](https://www.davidsilver.uk/teaching/) — David Silver (2015)
  Pages: Lecture 5: Model-Free Control
  伦敦大学学院备受推崇的系列讲座，清晰地解释了包括SARSA和Q-学习在内的基础强化学习算法。
