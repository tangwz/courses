# SARSA：同策略TD控制

来源：[原文](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/sarsa-on-policy-td-control)

[返回章节目录](README.md) · [返回课程目录](../README.md)

时序差分 (TD) 方法，如TD(0)，通过自举来估算给定策略$\pi$的状态价值函数$V_\pi$。这些方法根据观测到的奖励和下一个状态的*估计*价值来更新状态的价值估计。这使得学习能够从每一步中进行，与蒙特卡罗方法不同，后者需要等到一个完整的回合结束。

然而，如果我们的目标是*控制*，即找到最优策略，那么仅仅估算$V_\pi$是不够的。对于控制问题，特别是在没有环境模型的情况下，我们通常需要学习动作价值，$Q_\pi(s, a)$。这告诉我们，在状态$s$采取动作$a$之后，并随后遵循策略$\pi$所获得的预期回报。了解$Q$值使智能体能够在每个状态中，通过选择具有最高$Q$值的动作来选取最佳动作。

SARSA是为控制问题设计的一种TD方法。它直接估算动作价值函数$Q_\pi$。SARSA这个名称强调了其更新计算中涉及的事件顺序：智能体从一个**状**态（$S_t$）开始，选择一个**动**作（$A_t$），收到一个**奖**励（$R_{t+1}$），并转换到下一个**状**态（$S_{t+1}$），然后根据其当前策略选择下一个**动**作（$A_{t+1}$）。这个五元组$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$构成了SARSA更新的根本要素。

### SARSA 更新规则

与TD(0)一样，SARSA在每个时间步之后执行一次更新。动作价值函数$Q(S_t, A_t)$的更新规则如下：


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$


我们来分析一下：

- $Q(S_t, A_t)$ 是在状态$S_t$执行动作$A_t$的当前价值估计。
- $\alpha$ 是学习率（一个小的正数，例如0.1），它控制我们根据新信息更新估计的程度。
- $R_{t+1}$ 是在采取动作$A_t$后立即获得的奖励。
- $\gamma$ 是折扣因子（介于0和1之间），用于平衡即时奖励与未来奖励。
- $Q(S_{t+1}, A_{t+1})$ 是*下一个状态-动作对*的估计价值。这是自举部分。与蒙特卡洛方法等待最终回报不同，我们使用智能体将根据其策略采取的下一个动作的当前价值估计。
- $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ 是TD目标。它表示基于一步经验的$Q(S_t, A_t)$的更好估计。
- $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$ 是TD误差，衡量估计价值与TD目标之间的差异。

更新本质上是将当前估计值$Q(S_t, A_t)$微调 (fine-tuning)向TD目标靠近。这里的一个重要之处请注意：目标包含$Q(S_{t+1}, A_{t+1})$，它使用了智能体当前策略在状态$S_{t+1}$中实际选择的动作$A_{t+1}$。

### SARSA 控制算法

SARSA在一个通用的控制循环中使用这个更新规则，旨在随着时间推移改进策略。由于我们需要学习所有动作的价值以找到最佳动作，所遵循的策略必须能够进行新尝试。一种常见的方法是使用基于当前$Q$值估计的$\epsilon$-贪婪策略。这意味着智能体通常选择具有最高$Q$值的动作（利用），但偶尔也会以$\epsilon$的概率选择一个随机动作（进行尝试）。

SARSA算法的典型结构如下：

1. **初始化：**

   - 初始化所有状态-动作对$s \in \mathcal{S}, a \in \mathcal{A}(s)$的$Q(s, a)$，可以任意设置（例如，设为0或乐观值）。
   - 确保$Q(\text{终止状态}, \cdot) = 0$。
   - 设置学习率$\alpha > 0$和折扣因子$\gamma \in [0, 1]$。
   - 为$\epsilon$-贪婪策略设置尝试参数 (parameter)$\epsilon > 0$。
2. **回合循环：** 对每个回合重复：

   - 初始化起始状态$S$。
   - 使用从$Q$导出的策略（例如，$\epsilon$-贪婪）从状态$S$中选择第一个动作$A$。
   - **步骤循环：** 对回合内的每一步重复：
     - 执行动作$A$。观测到奖励$R$和下一个状态$S'$。
     - 使用从$Q$导出的策略（例如，$\epsilon$-贪婪）从状态$S'$中选择*下一个*动作$A'$。
     - 执行SARSA更新：
       
       $$
       Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]
       $$
       
     - 更新下一次迭代的状态和动作：$S \leftarrow S'$，$A \leftarrow A'$。
   - 持续直到$S$是终止状态。

### SARSA 为何是同策略？

SARSA被归类为一种*同策略*算法。这意味着它学习的是智能体\*当前遵循的策略$\pi$\*的价值函数$Q_\pi$。所遵循的策略包含了尝试步骤（例如，$\epsilon$-贪婪策略所采取的随机动作）。

其同策略特性直接源于更新规则：$Q(S_{t+1}, A_{t+1})$使用了当前策略在状态$S_{t+1}$中实际选择的动作$A_{t+1}$。算法根据智能体的实际行为，包括其尝试策略，来估计采取动作的价值。

这具有实际意义。如果智能体使用$\epsilon$-贪婪策略，SARSA会学习与该$\epsilon$-贪婪策略相关的价值。由此产生的$Q$值反映了考虑到智能体可能以$\epsilon$的概率采取随机的、次优动作时的预期回报。这通常会导致在尝试失误代价高昂（例如，从悬崖坠落）的环境中找到“更安全”的路径。智能体学习如何在进行尝试时考虑到自身可能出现的失误。

将其与蒙特卡洛控制方法进行对比，蒙特卡洛方法也根据所遵循的策略进行学习，但只在回合结束后才更新。SARSA提供了TD学习的好处（每一步都更新），同时仍然学习正在使用的策略。

在下一节中，我们将介绍Q-学习，一个密切相关的TD控制算法，它采取了不同的方法：它直接学习最优策略，而不考虑所遵循的尝试策略。这使得Q-学习成为一种*异策略*方法。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapter 6.4
  强化学习领域的权威教材，详细阐述了TD学习、动作价值函数和SARSA算法等基础知识。
- [On-line Q-learning using connectionist networks](https://www.researchgate.net/publication/235773822_On-Line_Q-Learning_Using_Connectionist_Systems/fulltext/553d712f0cf2d131f31f9e2e/On-Line-Q-Learning-Using-Connectionist-Systems.pdf) — Gavin Adrian Rummery, Mahesan Niranjan (1994)
  Journal: Technical Report CUED/F-INFENG/TR166; Publisher: Cambridge University Engineering Department
  这份技术报告首次提出了“修改版Q学习”，即SARSA算法，并详细描述了其在线策略TD控制方法。
- [RL Course by David Silver - Lecture 4: Model-Free Control](https://youtu.be/Nd1-UUMVfz4) — David Silver (2015)
  Publisher: Google DeepMind
  强化学习系列讲座之一，清晰地解释了无模型控制方法，包括SARSA及其在线策略特点。

---

[上一节](03-TD%E5%AD%A6%E4%B9%A0%E7%9B%B8%E5%AF%B9%E4%BA%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%96%B9%E6%B3%95%E7%9A%84%E4%BC%98%E5%8A%BF.md) · [下一节](05-Q%E5%AD%A6%E4%B9%A0%EF%BC%9A%E7%A6%BB%E7%AD%96%E7%95%A5TD%E6%8E%A7%E5%88%B6.md)
