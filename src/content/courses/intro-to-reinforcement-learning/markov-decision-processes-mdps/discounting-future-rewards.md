---
course: "intro-to-reinforcement-learning"
chapter: "markov-decision-processes-mdps"
lesson: "discounting-future-rewards"
sourceId: 1592
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps/discounting-future-rewards"
title: "未来奖励的折现"
description: "解释马尔可夫决策过程中的折现因子伽马 (\\gamma) 的作用和影响。"
order: 6
plots: ["plots/1592-0.json"]
sourceHash: "d629629b1de09b6af7502d62a0f07c879be059a3fa4d2abcf073367037fa5687"
sourceCorrections: []
---

“回报” $G_t$ 被定义为智能体从时间步 $t$ 开始预期积累的总奖励。对于有明确终止点的任务（回合制任务），这仅仅是从 $t+1$ 开始直到回合结束的所有奖励的总和。

然而，许多强化学习 (reinforcement learning)问题涉及没有自然终止点的持续互动（连续任务）。设想一个智能体管理电网或一个持续运行的交易机器人。如果我们只是无限期地累加奖励，回报 $G_t$ 很容易变为无限大。比较无限大的回报无法帮助我们判断哪种策略更好。此外，即使在回合制任务中，未来100步收到的奖励是否应与现在收到的奖励价值相同？

这就引出了**折现**的思想。我们引入一个参数 (parameter)，即折现因子，用希腊字母伽马 ($\gamma$) 表示，以系统地降低未来奖励的价值。

折现因子 $\gamma$ 是一个介于0和1之间的数字 ($0 \le \gamma \le 1$)。回报 $G_t$ 现在定义为*折现*未来奖励的总和：


$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$


我们来解析一下这个定义：

- 晚一步收到的奖励 $R_{t+1}$ 直接添加（乘以 $\gamma^0 = 1$）。
- 晚两步收到的奖励 $R_{t+2}$ 乘以 $\gamma$。
- 晚三步收到的奖励 $R_{t+3}$ 乘以 $\gamma^2$。
- 依此类推；晚 $k+1$ 步收到的奖励 $R_{t+k+1}$ 乘以 $\gamma^k$。

由于 $\gamma$ 小于或等于1，随着 $k$ 的增加，$\gamma^k$ 变得越来越小。这意味着未来更远时间收到的奖励对总折现回报 $G_t$ 的贡献较小。

### 为何折现？

折现有以下几个重要作用：

1. **数学上的便利性**：在连续任务中，折现确保定义 $G_t$ 的无限和收敛到一个有限值，前提是奖励有界。如果任何一步的奖励保证不大于某个值 $R_{max}$，那么回报 $G_t$ 保证小于或等于 $\frac{R_{max}}{1-\gamma}$（利用无限几何级数的求和公式）。这使得比较不同策略在数学上可行。
2. **模拟对近期奖励的偏好**：通常，尽早收到奖励比稍后收到相同奖励更可取。可以想一下金融利率或生物为即时生存而产生的本能。折现自然地模拟了这种偏好。
3. **应对不确定性**：未来本身就存在不确定性。智能体对环境的模型可能不完善，或者环境本身可能出乎意料地改变。应用折现因子隐含地考虑了随着时间推移不确定性的增加；遥远未来的奖励不确定性更高，因此价值较低。

### 伽马值 ($\gamma$) 的选择

$\gamma$ 的选择显著影响智能体的行为：

- **如果 $\gamma = 0$**：智能体变得完全'短视'。它只关心最大化即时奖励 $R_{t+1}$。它不顾其行动的所有未来影响。回报就是 $G_t = R_{t+1}$。
- **如果 $\gamma$ 接近1（例如 0.99）**：智能体具有'远见'。未来奖励的价值几乎与即时奖励一样高。智能体更侧重于其行动的长期影响。
- **如果 $\gamma$ 介于0和1之间**：智能体平衡即时奖励和未来奖励。较小的值优先考虑短期收益，而较大的值则强调长期累积。

$\gamma$ 的具体值通常被视为问题设置的超参数 (parameter) (hyperparameter)，根据任务的性质选择。例如，在金融应用中，$\gamma$ 可能与当前利率有关。在回合制任务中，如果回合保证终止，有时会使用 $\gamma=1$（无折现），尽管使用 $\gamma < 1$ 通常可以促成更快的学习。

下图展示了折现因子 $\gamma^k$ 如何对于不同的 $\gamma$ 值，在未来时间步 $k$ 上降低奖励的价值：



![折现因子 (γ) 随时间的影响](plots/1592-0.json)



> 未来 $k$ 步收到的奖励所赋权重 (weight)随着 $k$ 呈指数衰减。更高的 $\gamma$ 会导致衰减变慢，更多地重视未来奖励。

在强化学习 (reinforcement learning)中，最终目标通常是找到一个策略 $\pi$，使每个状态的*预期*折现回报最大化。因此，折现的思想对于定义我们试图实现的目标非常重要，并使得形式化方法成为可能，比如我们接下来会讨论的价值函数，这些方法使我们能够解决马尔可夫决策过程。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 3: Finite Markov Decision Processes
  强化学习领域的经典教科书，全面阐述了马尔可夫决策过程和折扣因子的作用。
- [Reinforcement Learning: Lecture 2 - Markov Decision Processes](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf) — David Silver (2015)
  Publisher: University College London (UCL)
  广受引用的系列讲座之一，本讲座介绍了马尔可夫决策过程，并解释了强化学习中折扣的概念和目的。
