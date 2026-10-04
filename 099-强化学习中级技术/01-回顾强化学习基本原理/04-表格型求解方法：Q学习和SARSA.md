# 表格型求解方法：Q学习和SARSA

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/tabular-methods-review)

[返回章节目录](README.md) · [返回课程目录](../README.md)

免模型方法在强化学习 (reinforcement learning)中指的是那些能够直接从经验中学习价值函数，而无需预定义环境动态（状态转移和奖励）模型的算法。这些方法在马尔可夫决策过程（MDP）及其相关价值函数的框架内运行。其中最基本的是Q学习和SARSA，两者都属于时序差分（TD）学习算法的范畴。它们根据其他习得的估计值来更新价值估计，从而从当前价值函数进行自举。当状态空间和动作空间足够小，能够用表格表示时，这些方法能有效运行，因此被称为“表格型方法”。

### Q学习：异策略地学习最优价值函数

Q学习是一种广受欢迎的异策略TD控制算法。其目的是直接习得最优动作价值函数，$Q^*(s, a)$，它表示从状态$s$开始，采取动作$a$，并在此之后遵循最优策略$\pi^*$所获得的预期回报。所谓“异策略”，意味着即使在学习过程中采取的动作是由不同且可能带有探索性的策略（例如epsilon-greedy）确定的，Q学习也能习得最优价值函数。

智能体与环境交互，观察状态转移$(S_t, A_t, R_{t+1}, S_{t+1})$。在每次状态转移后，它使用以下规则更新其对$Q(S_t, A_t)$的估计：


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]
$$


我们来分析一下这个更新公式：

- $Q(S_t, A_t)$: 在状态$S_t$下采取动作$A_t$的当前估计价值。
- $\alpha$: 学习率（一个较小的值，例如0.1），控制新信息在多大程度上取代旧估计值。
- $R_{t+1}$: 采取动作$A_t$后获得的奖励。
- $\gamma$: 折扣因子（介于0和1之间），用于平衡即时奖励与未来奖励。
- $\max_{a'} Q(S_{t+1}, a')$: Q学习的核心部分。这是从下一状态$S_{t+1}$可获得的最优未来价值的估计值。它会查看状态$S_{t+1}$中所有可能的后续动作$a'$的Q值，并选取最大值。这种针对*后续*动作的贪婪选择，使得Q学习能直接逼近最优$Q^*$。

项$R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$被称为*TD目标*。TD目标与当前估计值$Q(S_t, A_t)$之间的差值是*TD误差*，它推动了学习更新。

为了确保充分的探索，Q学习通常与epsilon-greedy（ε-贪婪）等策略结合使用，其中智能体以$\epsilon$的概率选择一个随机动作，并以$1-\epsilon$的概率选择当前Q值最高的动作（$argmax_a Q(S_t, a)$）。重要的是，更新规则使用了`max`操作符，它习得的是贪婪（最优）动作，即使实际执行的是一个探索性动作。

> 图示了Q学习单次更新步骤的信息流。

### SARSA：同策略地学习价值函数

SARSA（State-Action-Reward-State-Action，状态-动作-奖励-状态-动作）是另一种TD控制算法，它是*同策略*的。与Q学习不同，SARSA学习的是与当前遵循的*策略*相对应的动作价值函数$Q(s, a)$，包括其探索行为。

更新过程需要知道当前策略在下一状态$S_{t+1}$中选择的*后续动作* $A_{t+1}$。五元组$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$为该算法命名。更新规则为：


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$


请注意与Q学习的主要区别：SARSA没有对所有可能的后续动作取最大Q值（$\max_{a'} Q(S_{t+1}, a')$），而是使用智能体根据当前策略实际*将要*采取的特定后续动作$A_{t+1}$的Q值（例如，由epsilon-greedy策略为状态$S_{t+1}$选择的动作）。

这使得SARSA对正在执行的策略敏感。如果策略具有探索性（$\epsilon$值较高），SARSA会学习到考虑到采取次优探索性动作可能性的Q值。这可能导致在学习过程中，与Q学习相比，智能体表现得更“保守”，因为Q学习总是假设在下一步中将采取最优动作作为其更新目标。

> 图示了SARSA单次更新步骤的信息流，需要后续动作$A_{t+1}$。

### Q学习与SARSA：快速比较

- \*\*策略类型：\*\*Q学习是异策略；SARSA是同策略。
- \*\*更新目标：\*\*Q学习使用下一状态的最大Q值（$\max_{a'} Q(S', a')$），直接目标是最优价值。SARSA使用实际采取的后续动作的Q值（$Q(S', A')$），评估当前策略。
- \*\*行为表现：\*\*Q学习即使在次优探索时也能习得最优策略。SARSA习得的价值反映了探索策略；如果探索有时导致负面结果（例如在网格中掉下悬崖），它可能会学习到更安全的路径，而Q学习可能会习得假设它能避开悬崖的价值，即使探索策略有时会失败。

Q学习和SARSA都是许多高级强化学习 (reinforcement learning)技术的重要构成部分。它们通过迭代更新存储在表格中的价值估计值来工作，这些估计值以状态-动作对作为索引。然而，当处理状态或动作数量非常大或无限（例如连续空间）的问题时，它们对这种显式表格的依赖成为一个主要瓶颈，这也是下一节的主题。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  涵盖强化学习基本原理的综合教材，包含Q-Learning和SARSA的详细说明。
- [Learning from Delayed Rewards](https://www.repository.cam.ac.uk/bitstream/handle/1810/241513/Watkins_PhD_1989.pdf) — Christopher J.C.H. Watkins (1989)
  Publisher: University of Cambridge
  引入Q-learning算法的原始博士论文。
- [On-line Q-learning using connectionist systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGCoWYRtGAjO-oVACSaf9PGFqZ0wMAfhOQrxfyq8QQN7vt3okx3VrAQ3k2eg7EC9duECZEQeL1VucKeVm4hHI5b_z3ALWlTozDHINjl5-dLZNdBz5Aqk8-DJouzpaG_BaMoVm3BY9lyShmyZ33LR9O8i5Q9BwsrNbUWFAHt2KBt7FP3o-qiUZZ_OXWksWRayFEqeVuIq7BF_b4=) — Gavin A. Rummery, Mahesan Niranjan (1994)
  Journal: Technical Report CUED/F-INFENG/TR 166; Publisher: Cambridge University Engineering Department
  介绍SARSA算法的技术报告。

---

[上一节](03-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0%E4%B8%8E%E8%B4%9D%E5%B0%94%E6%9B%BC%E6%96%B9%E7%A8%8B.md) · [下一节](05-%E8%A1%A8%E6%A0%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
