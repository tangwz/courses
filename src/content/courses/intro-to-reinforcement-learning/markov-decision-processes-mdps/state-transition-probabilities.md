---
course: "intro-to-reinforcement-learning"
chapter: "markov-decision-processes-mdps"
lesson: "state-transition-probabilities"
sourceId: 1583
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps/state-transition-probabilities"
title: "状态转移概率"
description: "理解状态转移概率如何建模环境的动态特性。"
order: 3
plots: []
sourceHash: "7e2a83aee3b3f5c4fcddca689eb36da3921b2f4a3b472e4be2a95344d16307f3"
sourceCorrections: []
---

在确定了MDP的主要组成部分：状态 ($S$)、动作 ($A$)、奖励 ($R$) 和折扣因子 ($\gamma$) 之后，我们现在关注描述环境行为的元素：**状态转移概率**，通常表示为 $P$。这些概率定义了环境的*动态特性*，规定了状态如何响应智能体的动作而变化。

可以将转移概率看作支配环境的基本规则或物理原理。当智能体处于特定状态 $s$ 并选择执行动作 $a$ 时，接下来会发生什么？在许多情况下，结果并非固定不变。环境可能会响应并进入几个可能的下一个状态 $s'$ 中的一个。转移概率函数为我们提供了每个可能结果的发生概率。

正式地，状态转移概率定义为：


$$
P(s' | s, a) = \text{概率}\{ S_{t+1} = s' | S_t = s, A_t = a \}
$$


这个公式表示：“在时间步 $t$ 时当前状态是 $s$ 并且智能体执行动作 $a$ 的条件下，在下一个时间步 ($t+1$) 转移到状态 $s'$ 的概率。”

需要理解的是，这些概率描述了环境的动态特性，而不是智能体的决策过程（决策过程由策略 $\pi$ 支配）。对于从给定状态 $s$ 执行的任何动作 $a$，转移到*所有可能*的下一个状态 $s'$ 的概率总和必须为1：


$$
\sum_{s' \in S} P(s' | s, a) = 1, \quad \text{对于所有 } s \in S, a \in A(s)
$$


其中 $A(s)$ 是状态 $s$ 中可用的动作集合。

### 马尔可夫属性

MDPs 的一个基本假设是**马尔可夫属性**。此属性表明未来只取决于当前，而不取决于过去。在状态转移的背景下，这意味着移动到下一个状态 $s'$ 的概率*只*取决于当前状态 $s$ 和当前动作 $a$。在到达状态 $s$ 之前访问过的状态和执行过的动作的历史对于预测即时未来无关。

数学上，马尔可夫属性意味着：


$$
P(S_{t+1} = s' | S_t = s, A_t = a) = P(S_{t+1} = s' | S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0)
$$


“这种简化很有用，因为它使我们能够建模复杂的序列问题，而无需跟踪整个历史。如果当前状态 $s$ 充分捕获了预测未来所需的所有来自过去的相关信息，则马尔可夫属性成立。通过仔细定义状态表示以满足此属性，许多问题都可以有效地建模或近似为MDPs。”

### 例子：一个简单网格环境

考虑一个简单的3x3网格，其中智能体可以向北、南、东或西移动。令状态为坐标 $(x, y)$，其中 $x, y \in \{0, 1, 2\}$。

- **确定性转移：** 在一个完全可预测的环境中，从状态 $(0, 0)$ 执行动作“东”将总是导致状态 $(1, 0)$。转移概率将是 $P((1, 0) | (0, 0), \text{东}) = 1$，而最终到达任何其他状态的概率将是0。
- **随机性转移：** 现在，想象一个“湿滑的”网格。如果智能体从状态 $(0, 0)$ 选择动作“东”，可能它有80%的几率按预期工作（最终到达 $(1, 0)$），有10%的几率滑倒并停留在 $(0, 0)$，以及有10%的几率侧滑并最终到达状态 $(0, 1)$（假设 $(0, 1)$ 在 $(0, 0)$ 的北方）。转移概率将是：

  - $P((1, 0) | (0, 0), \text{东}) = 0.8$
  - $P((0, 0) | (0, 0), \text{东}) = 0.1$
  - $P((0, 1) | (0, 0), \text{东}) = 0.1$
  - $P(s' | (0, 0), \text{东}) = 0$ 对于所有其他状态 $s'$。

请注意 $0.8 + 0.1 + 0.1 = 1.0$。环境引入了不确定性。

这里有一个小图表，说明了在状态 (0,0) 采取“东”动作时的这些随机转移：

> 从状态 $s=(0,0)$ 在随机网格环境中执行动作 $a=\text{东}$ 时的转移概率。

### 已知模型与学习模型

在某些问题中，我们可能会明确地给出转移概率 $P(s' | s, a)$。这意味着我们拥有环境动态特性的完整*模型*。像动态规划（我们稍后会讨论）这样的技术依赖于拥有这样的模型。

然而，在许多实际的强化学习 (reinforcement learning)场景中，我们事先*不*知道这些概率。智能体必须纯粹通过互动，通过尝试动作和观察结果来采样转移 $(s, a, r, s')$，从而了解环境的动态特性。这是Q-learning和SARSA等*无模型*RL方法的范围，它们是本课程的主要议题。

理解状态转移概率对于掌握MDPs如何形式化序列决策问题是非常重要的。它们表示智能体必须应对的固有动态特性，无论这些动态特性是事先已知还是必须通过经验学习。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  强化学习的参考书，详细介绍了马尔可夫决策过程、状态转移和马尔可夫属性。
- [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) — Stuart Russell and Peter Norvig (2020)
  Publisher: Pearson
  一本全面的人工智能教材，其中一章专门介绍马尔可夫决策过程，解释了状态转移概率和马尔可夫属性。
- [Stanford CS234: Reinforcement Learning](https://cs234.stanford.edu/) — Emma Brunskill and Stanford University Staff (2025)
  Publisher: Stanford University
  斯坦福大学强化学习课程的讲座材料，特别是关于MDP的讲座，提供了关于马尔可夫决策过程的清晰学术视角。
