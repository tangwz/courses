# Q学习：离策略TD控制

来源：[原文](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/q-learning-off-policy-td-control)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Q学习由Chris Watkins于1989年提出，是一种重要的离策略时序差分（TD）控制算法。与SARSA等学习智能体当前遵循策略（包括其试探步骤）价值的算法不同，Q学习提供了一种独特且通常更直接的方法。

在当前语境中，“离策略”意味着什么？它表示所要学习的策略（目标策略）与用于产生行为的策略（行为策略）不同。Q学习直接针对最优动作-价值函数，$Q^*(S, A)$，无论智能体实际使用何种策略来对环境进行尝试。Q学习中的目标策略是相对于当前动作-价值估计的贪婪策略，而行为策略可以是更具试探性的，例如epsilon-贪婪策略。

### Q学习更新规则

Q学习的核心在于其更新规则。在状态$S_t$中采取动作$A_t$并观察到奖励$R_{t+1}$和下一状态$S_{t+1}$之后，Q学习使用以下规则更新其估计值$Q(S_t, A_t)$：


$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$


我们来分解一下：

1. $Q(S_t, A_t)$：所经历的状态-动作对的当前动作-价值估计。
2. $\alpha$：学习率，决定新估计值对旧估计值的影响程度。
3. $R_{t+1}$：执行动作$A_t$后立即获得的奖励。
4. $\gamma$：折扣因子，使当前奖励比未来奖励更有价值。
5. $\max_{a} Q(S_{t+1}, a)$：**这是Q学习的决定性部分。** Q学习不是使用实际采取的下一个动作（$A_{t+1}$，如SARSA中所示）的Q值，而是使用下一状态$S_{t+1}$中所有可用动作$a$的最大可能Q值。它假定智能体从状态$S_{t+1}$开始遵循最优（贪婪）策略，并以此估计回报。
6. $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$：这是TD目标，表示从状态$S_t$开始并采取动作$A_t$的预期回报的更新估计。
7. $[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) ]$：这是TD误差，是新估计回报（TD目标）与旧估计值$Q(S_t, A_t)$之间的差。

通过对下一动作取最大值，Q学习能够学习贪婪策略（该策略总是选择使$Q(S_{t+1}, a)$最大的动作`a`），即使智能体行为不同（例如，偶尔通过epsilon-贪婪策略采取随机动作进行尝试）。

> SARSA和Q学习在TD目标计算中所用信息的比较。SARSA使用实际的下一个状态-动作对（$S_{t+1}, A_{t+1}$）的Q值，而Q学习则使用下一个状态$S_{t+1}$中所有可能动作$a$的最大Q值。

### Q学习算法

以下是Q学习算法的伪代码：

```
初始化 Q(s, a) 为任意值，对于所有 s 属于 S, a 属于 A(s)
（例如，Q(s, a) = 0）
初始化 Q(终止状态, .) = 0

循环每个回合:
    初始化 S (回合的起始状态)
    循环回合的每个步骤:
        根据 Q 导出的策略选择动作 A (例如，epsilon-贪婪策略)
        执行动作 A，观察 R, S'

        # Q学习核心更新
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]

        S <- S'  # 移动到下一状态

        如果 S 是终止状态，跳出内循环 (回合结束)
结束循环 (回合)
```

**算法步骤：**

1. **初始化：** 将Q表用任意值初始化（通常为零）。终止状态的Q值必须为零。设置学习参数 (parameter)$\alpha$（学习率）和$\gamma$（折扣因子），以及试探策略的参数（例如epsilon-贪婪策略的$\epsilon$）。
2. **回合循环：** 重复若干个回合。
3. **状态初始化：** 从回合的初始状态$S$开始。
4. **步骤循环：** 重复直到当前状态$S$是终止状态。
5. **动作选择：** 根据当前状态$S$和当前Q值选择动作$A$。epsilon-贪婪策略很常见：以$1-\epsilon$的概率选择Q值最高的动作（$\arg\max_a Q(S, a)$），以$\epsilon$的概率选择一个随机动作。这是*行为策略*。
6. **执行与观察：** 执行动作$A$，观察奖励$R$和下一状态$S'$。
7. **Q值更新：** 使用$S, A, R, S'$应用Q学习更新规则。再次注意，更新使用了$\max_a Q(S', a)$，这代表了*目标策略*（贪婪策略）。
8. **状态转换：** 更新当前状态：$S \leftarrow S'$。
9. **终止检查：** 如果$S'$是终止状态，则回合结束。

Q学习在标准随机逼近条件下被证明可以收敛到最优动作-价值函数$Q^*$，前提是所有状态-动作对都能持续被访问。使用$\epsilon$随时间衰减的epsilon-贪婪行为策略，能确保初期有充分的试探，并最终收敛到对所学最优策略的使用。

由于Q学习直接学习最优Q函数，与SARSA等在策略方法相比，它通常被认为在找到最优策略方面样本效率更高，尤其是在试探范围广的情况下。然而，这种对最优性的直接追求意味着它不考虑与试探策略本身相关的成本或风险，这在安全要求高的应用中可能需要考虑，在此类应用中，SARSA在学习阶段可能更受青睐。

## 参考资料

- [Learning from Delayed Rewards](https://www.cs.rhul.ac.uk/home/chrisw/new_thesis.pdf) — Christopher John Cornish Hellaby Watkins (1989)
  Publisher: University of Cambridge
  这篇博士论文首次提出了Q-learning算法，确立了离策略时序差分控制的原则。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapter 6: Temporal-Difference Learning, especially Section 6.5: Q-Learning: Off-policy TD Control
  强化学习的标准教科书，对Q-Learning、其更新规则、算法和理论基础进行了全面解释。
- [UCL Course on Reinforcement Learning - Lecture 6: Off-policy Control](https://github.com/zyxue/youtube_RL_course_by_David_Silver) — David Silver (2015)
  Publisher: University College London (UCL); Pages: Relevant sections on Q-Learning (e.g., slides 6-10)
  广受推崇的大学课程讲义，简洁地解释了离策略控制和Q-Learning算法。

---

[上一节](04-SARSA%EF%BC%9A%E5%90%8C%E7%AD%96%E7%95%A5TD%E6%8E%A7%E5%88%B6.md) · [下一节](06-%E6%AF%94%E8%BE%83%20SARSA%20%E4%B8%8E%20Q-%E5%AD%A6%E4%B9%A0.md)
