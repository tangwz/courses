---
course: "intro-to-reinforcement-learning"
chapter: "markov-decision-processes-mdps"
lesson: "modeling-sequential-decision-making"
sourceId: 1576
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps/modeling-sequential-decision-making"
title: "序贯决策建模"
description: "介绍对具有长期影响的决策问题进行建模的挑战。"
order: 1
plots: []
sourceHash: "32875a47c53bb080bfe998148b7ac9aa9b5e2054ec6cf78427d84a722be30256"
sourceCorrections: []
---

在上一章中，我们建立了强化学习 (reinforcement learning)中基本的交互循环：智能体观察状态，采取行动，获得奖励，并转移到新状态。这个循环重复，智能体的目标通常是使随时间积累的总奖励最大化。

然而，许多我们希望通过强化学习解决的问题都涉及一系列决策，其中做出的选择会产生后续影响。例如，教机器人导航建筑物，训练算法下围棋，或者优化供应链中的库存管理。这些问题有几个共同点：

- **序贯性：** 决策并非孤立进行。当前采取的行动直接影响智能体接下来遇到的情境（状态），可能产生或限制未来的机会。在十字路口左转意味着你不会立即看到右边路径的情况。
- **延迟后果：** 一个行动的影响可能不会立即完全显现。短期收益可能导致长期损失，反之亦然。国际象棋中弃子可能获得只有在多步之后才能体现的局面优势。相反，追求即时利润最大化可能耗尽未来发展所需的资源。
- **随机性：** 环境对行动的响应可能涉及随机性。请求网络资源可能因网络状况而成功或失败。机器人前进的指令可能因轮子打滑而导致实际移动略有不同。游戏中的对手通过其行动引入不确定性。

为了有效应对这些复杂情况，开发智能体，我们需要的不仅仅是基本的智能体-环境循环的思路。我们需要一种正式的方式来描述问题本身，包括智能体可能处于的状态、可以采取的行动、状态如何响应行动而变化（环境的动态），以及在此过程中获得的奖励。这种正式描述使我们能够严谨地思考问题，并开发出能够平衡即时奖励与长期目标的优化策略的算法。

考虑一个简化的房间导航示意：

> 智能体需要从房间 A 到达房间 D。从 A 向北走通常会到达 B，但有时（概率为 0.2）会到达一个危险区域（房间 C）。行动具有概率性结果，并导致不同的后续状态和潜在奖励。

在这个场景中，智能体的位置代表着*状态*。可用的*行动*取决于状态（例如，“向北走”、“向东走”）。环境的动态由*转移概率*（例如从房间 A 向北移动时的 80%/20% 分布）体现。到达目标会产生积极的*奖励*，而进入危险区域可能会获得负面奖励。

这种将不确定性下的序贯决策问题正式结构化的必要性直接引出了**马尔可夫决策过程（MDPs）**。MDPs 提供了强化学习中普遍使用的标准数学框架。它们提供了一种精确的方式来定义环境的组成部分和交互，从而使得学习算法的开发和分析成为可能。在接下来的章节中，我们将分解 MDP 的正式定义及其核心构成：状态、行动、转移概率、奖励和折扣因子。理解 MDPs 对于理解强化学习智能体如何在复杂、动态的环境中学习最优行为是重要的。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本广受认可的教材，全面介绍了强化学习，并在其早期章节详细讲解了马尔可夫决策过程。
- [CS234: Reinforcement Learning (Spring 2023)](https://cs234.stanford.edu/) — Emma Brunskill (2025)
  Journal: Stanford University; Publisher: Stanford University
  斯坦福大学的官方课程材料，提供了强化学习的讲座和资源，其中包含马尔可夫决策过程的基础内容。
- [Dynamic Programming and Optimal Control (Vol. 1)](http://www.athenasc.com/dpbook.html) — Dimitri P. Bertsekas (2017)
  Publisher: Athena Scientific
  一本严谨且权威的动态规划教材，为解决序贯决策问题（包括马尔可夫决策过程）提供了数学基础。
