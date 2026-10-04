---
course: "intro-to-reinforcement-learning"
chapter: "monte-carlo-methods"
lesson: "on-policy-first-visit-mc-control"
sourceId: 1638
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-4-monte-carlo-methods/on-policy-first-visit-mc-control"
title: "同策略首次访问蒙特卡洛控制的实现"
description: "概述同策略首次访问蒙特卡洛控制的算法。"
order: 6
plots: []
sourceHash: "d09dd8b88593aef4cdadd44e0d7d859fab5888b33d893157a87dbf87aeccf05f"
sourceCorrections: []
---

好的，我们现在将蒙特卡洛预测的思想与试探需求结合起来，构建一个控制算法。请记住，强化学习 (reinforcement learning)控制的目标不仅是评估给定策略，而是找到一个*最优*策略。在蒙特卡洛方法的背景下，这意味着找到一个能根据从回合中收集到的经验使预期回报最大化的策略。

我们将着重介绍一种*同策略*方法，即我们学习的策略($\pi$)与用于生成行为（即回合）的策略是相同的。具体来说，我们将详细说明*首次访问蒙特卡洛控制*算法。顾名思义，它仅基于状态-动作对$(s, a)$在回合中*首次*访问后的回报来更新其价值估计。

### 蒙特卡洛控制的难点：估计动作价值

为了改进策略，我们通常需要动作价值$Q^\pi(s, a)$的估计值，而不仅仅是状态价值$V^\pi(s)$。为什么？因为要判断在状态$s$中改变动作是否有益，我们需要知道从该状态$s$采取替代动作$a'$的预期回报。仅仅知道$V^\pi(s)$是不够的。

因此，我们的蒙特卡洛预测任务从估计$V^\pi(s)$转变为估计$Q^\pi(s, a)$。核心思想不变：对访问状态-动作对后观测到的回报进行平均。对于*首次访问蒙特卡洛*，我们平均每个回合中$(s, a)$首次出现之后的回报。

Q价值估计的更新规则与V价值估计非常相似：

1. 任意初始化所有$s \in \mathcal{S}, a \in \mathcal{A}(s)$的$Q(s, a)$。
2. 初始化$N(s, a) \leftarrow 0$（用于首次访问$(s,a)$次数的计数器）。
3. 初始化Returns$(s, a) \leftarrow$ 一个空列表（用于存储回报）。
4. 无限循环（针对每个回合）：
   - 遵循当前策略$\pi$生成一个回合：$S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$。
   - 初始化$G \leftarrow 0$。
   - 针对每个时间步$t = T-1, T-2, \dots, 0$循环：
     - $G \leftarrow \gamma G + R_{t+1}$（更新回报）。
     - 除非对$(S_t, A_t)$出现在$S_0, A_0, \dots, S_{t-1}, A_{t-1}$中：
       - 增加计数器：$N(S_t, A_t) \leftarrow N(S_t, A_t) + 1$。
       - 通过平均更新动作价值估计：

       
       $$
       Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} (G - Q(S_t, A_t))
       $$
       

       - （或者，将$G$添加到Returns$(S_t, A_t)$并重新计算$Q(S_t, A_t)$为该列表的平均值）。

这个过程为我们提供了使用蒙特卡洛的*策略评估*部分。那么，我们如何实现*策略改进*呢？

### 策略改进与试探问题

类似于动态规划，我们可以通过使策略相对于当前动作价值估计是贪婪的来改进策略。对于给定状态$s$，改进后的策略将选择使$Q(s, a)$最大化的动作$a$：


$$
\pi'(s) = \underset{a}{\text{argmax}} Q(s, a)
$$


然而，如果我们*仅仅*根据当前估计采取贪婪行动，就会有一个重要的限制。蒙特卡洛方法从遵循当前策略生成的回合中学习。如果策略在早期变得确定性且贪婪，它可能会完全停止对某些动作的试探。如果在状态$s$中某个动作$a$从未被选择，我们将永远不会获得对$(s, a)$对的回报，并且我们的$Q(s, a)$估计值永远不会改进。我们可能会收敛到次优策略，仅仅因为我们试探不足以找到真正的最优动作。

### 确保试探：$\epsilon$-贪婪策略

为了平衡试探（尝试新动作）与利用（使用已知最佳动作），我们使用确保持续试探的策略。一种常见且简单的方法是$\epsilon$-贪婪策略。

一个$\epsilon$-贪婪策略大部分时间表现为贪婪的，但会以小概率$\epsilon$选择一个随机动作。

- 以$1 - \epsilon$的概率，选择当前具有最高估计Q值的动作$a$：$\underset{a}{\text{argmax}} Q(s, a)$。
- 以$\epsilon$的概率，从状态$s$中所有可用动作中随机选择一个动作，选择时采用均匀概率。

这确保了，最终每个状态-动作对都有被访问的非零概率，使得我们的Q价值估计能够持续改进所有动作。我们学习的策略和我们用来生成回合的策略都是这种$\epsilon$-贪婪策略，因此称之为“同策略”。

### 同策略首次访问蒙特卡洛控制算法

现在我们可以将Q值的蒙特卡洛估计与$\epsilon$-贪婪策略改进结合起来。这遵循广义策略迭代（GPI）的模式，评估和改进步骤在此过程中相互影响。

**算法：同策略首次访问蒙特卡洛控制（针对$\epsilon$-软策略）**

1. **初始化：**
   - 任意初始化所有$s \in \mathcal{S}, a \in \mathcal{A}(s)$的$Q(s, a) \in \mathbb{R}$。
   - 初始化所有$s \in \mathcal{S}, a \in \mathcal{A}(s)$的$N(s, a) \leftarrow 0$。
   - 基于Q初始化一个$\epsilon$-贪婪策略$\pi$。对于任意状态$s$：
     - 对于所有$a \in \mathcal{A}(s)$，$\pi(a|s) \leftarrow \epsilon / |\mathcal{A}(s)|$。
     - $a^* \leftarrow \underset{a}{\text{argmax}} Q(s, a)$（任意处理平局）。
     - $\pi(a^*|s) \leftarrow \pi(a^*|s) + (1 - \epsilon)$。
2. **无限循环（针对每个回合）：**
   - **(a) 生成回合：** 使用当前策略$\pi$生成一个回合$S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$。
   - **(b) 初始化回报：** $G \leftarrow 0$。
   - **(c) 针对每个时间步$t = T-1, T-2, \dots, 0$循环：**
     - 更新回报：$G \leftarrow \gamma G + R_{t+1}$。
     - 检查首次访问：除非对$(S_t, A_t)$出现在$S_0, A_0, \dots, S_{t-1}, A_{t-1}$中：
       - 增加计数器：$N(S_t, A_t) \leftarrow N(S_t, A_t) + 1$。
       - 更新Q价值估计（策略评估）：
         
         $$
         Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} (G - Q(S_t, A_t))
         $$
         
       - 更新策略（策略改进）：使策略$\pi$相对于更新后的$Q$对状态$S_t$是$\epsilon$-贪婪的：
         - 重新计算概率：对于所有$a \in \mathcal{A}(S_t)$，$\pi(a|S_t) \leftarrow \epsilon / |\mathcal{A}(S_t)|$。
         - 找到最佳动作：$a^* \leftarrow \underset{a}{\text{argmax}} Q(S_t, a)$。
         - 更新最佳动作的概率：$\pi(a^*|S_t) \leftarrow \pi(a^*|S_t) + (1 - \epsilon)$。

> 同策略首次访问蒙特卡洛控制的交互循环。使用当前$\epsilon$-贪婪策略生成的的回合用于更新Q值（评估），而Q值又反过来用于更新$\epsilon$-贪婪策略（改进）。

该算法在每个回合的处理中，逐步交替进行评估和改进。它保证试探持续进行，因为策略$\pi$保持$\epsilon$-软（所有动作都有非零概率）。随着收集的回合增多，Q价值估计收敛于$\epsilon$-贪婪策略$\pi$的真实动作价值$Q^\pi(s, a)$，并且策略本身逐步改进以趋近最优$\epsilon$-贪婪策略。虽然不一定是真正的最优策略（由于强制试探），但它通常很接近，并且明显优于随机行为。

实际中，您可以使用Python中的字典来存储$Q(s, a)$和$N(s, a)$，其中键是状态-动作元组`(s, a)`。回合生成涉及根据动态存储或计算的当前$\epsilon$-贪婪策略概率来采样动作。随后的实践部分将提供一个更具体的例子。

## 参考资料

- [Reinforcement Learning: An Introduction](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH1BK9tZLPdymvl43IIipjggk98ptt5bXgdhOcuTGApfQQdDReaLycEhQg3IxtWABloIUy-nIgaD_EiOuPDw3OHzQPudKi3Ch_MRd9rhcDlMXUfnX2R7MHtXWCb-tmERLDPEYWHPG-O-GtMAA==) — Richard S. Sutton, Andrew G. Barto (2018)
  Publisher: The MIT Press
  涵盖蒙特卡洛方法、在策略控制和ε-贪婪策略的综合性标准教材。
- [Algorithms for Reinforcement Learning](http://www.morganclaypool.com/doi/abs/10.2200/S00268ED1V01Y201005AIM009) — Csaba Szepesvári (2010)
  Publisher: Morgan & Claypool Publishers; Volume: 9; Pages: 104; DOI: [10.2200/S00268ED1V01Y201005AIM009](https://doi.org/10.2200/S00268ED1V01Y201005AIM009)
  一本简洁的专题论著，提供了包括蒙特卡洛控制方法在内的强化学习算法的理论处理。
