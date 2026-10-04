---
course: "intro-to-reinforcement-learning"
chapter: "monte-carlo-methods"
lesson: "monte-carlo-control"
sourceId: 1628
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-4-monte-carlo-methods/monte-carlo-control"
title: "蒙特卡洛控制：估算 Qπ"
description: "将MC预测扩展到控制，通过估算动作价值函数（Qπ）。"
order: 3
plots: []
sourceHash: "eedad66d7e5e12f6d53b2fe5e71940c680216696fdf0dda62d0c4749ab1bcf9b"
sourceCorrections: []
---

蒙特卡洛（MC）方法通过平均访问状态 $s$ 后观察到的回报，可以估算给定策略 $\pi$ 的状态价值函数 $V^\pi(s)$。这个过程被称为MC预测或评估，它告诉我们在当前策略下处于特定状态有多好。然而，强化学习 (reinforcement learning)的最终目的通常是*控制*——找到使所有状态的期望回报最大化的*最优*策略 $\pi^*$。

为了使用MC方法实现控制，特别是当我们没有环境模型（即不知道转移概率 $p(s', r | s, a)$）时，估算状态价值函数 $V^\pi$ 通常是不够的。为什么？因为改进策略需要了解在给定状态下选择不同*动作*的价值。如果我们只知道 $V^\pi(s)$，在不知道环境动态的情况下，我们无法轻易判断哪个动作 $a$ 能带来最好的结果。我们将需要模型来计算 $\sum_{s', r} p(s', r | s, a) [r + \gamma V^\pi(s')]$。

这就是动作价值函数 $Q^\pi(s, a)$ 变得非常重要的地方。回想一下，$Q^\pi(s, a)$ 表示从状态 $s$ 开始，采取动作 $a$，然后遵循策略 $\pi$ 所能获得的期望回报。如果我们对状态 $s$ 中所有可用动作 $a$ 都有准确的 $Q^\pi(s, a)$ 估算，那么选择最佳动作就简单直接了：我们只需选择具有最高估算 $Q$ 值的动作。

$\pi'(s) = \underset{a}{\text{argmax}} Q(s, a)$

因此，对于使用MC方法的免模型控制，我们将重心从估算 $V^\pi(s)$ 转向估算 $Q^\pi(s, a)$。

### 使用蒙特卡洛估算动作价值（$Q^\pi$）

核心思想与MC预测保持一致：我们从完整的经验片段中学习，并平均观察到的回报。然而，我们不再是平均每个访问状态的回报，而是平均每个访问的*状态-动作对* $(s, a)$ 的回报。

我们来看看应用于估算 $Q^\pi$ 的首次访问MC方法。我们运行多个遵循策略 $\pi$ 的片段。对于每个片段，我们查看发生的每个状态-动作对 $(S_t, A_t)$。如果这是该特定对 $(S_t, A_t)$ 在此片段中*首次*被访问，我们从该点到片段结束计算总回报 $G_t$：

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T$

其中 $T$ 是片段的终止时间步。

我们为每个状态-动作对保持一个估算 $Q(s, a)$。与 $V^\pi$ 的MC预测类似，我们可以通过平均所有片段中首次访问 $(S_t, A_t)$ 后观察到的回报 $G_t$ 来更新我们的估算 $Q(S_t, A_t)$。

实现此方法的一个实用方式是使用增量更新规则。我们记录每个状态-动作对 $(s, a)$ 在所有片段中首次被访问的次数，称此计数为 $N(s, a)$。在生成一个片段并计算首次访问 $(S_t, A_t)$ 后的回报 $G_t$ 后，我们如下更新我们的估算 $Q(S_t, A_t)$：

1. 增加计数：$N(S_t, A_t) \leftarrow N(S_t, A_t) + 1$
2. 更新估算：
   $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} (G_t - Q(S_t, A_t))$

此公式向新观察到的回报 $G_t$ 的方向修正当前估算 $Q(S_t, A_t)$。步长 $\frac{1}{N(S_t, A_t)}$ 保证我们实际上是在计算一个累积平均值；早期回报影响较大，而随着估算变得更可靠，后期回报引起的调整较小。通常，对于非平稳问题（环境或策略可能发生变化），会使用恒定步长 $\alpha$ 代替 $\frac{1}{N(s, a)}$：

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$

其中 $0 < \alpha \le 1$ 是学习率参数 (parameter)。

### 试探的挑战

现在我们有了估算给定策略 $\pi$ 的动作价值 $Q^\pi(s, a)$ 的方法。但我们如何使用它来找到*最优*策略 $\pi^*$？

自然的想法是遵循广义策略迭代（GPI）的模式，它涉及在策略评估（估算当前策略的价值函数）和策略改进（使策略相对于当前价值函数估算变为贪婪策略）之间交替进行。

使用MC方法，这看起来会像：

1. **策略评估**：在策略 $\pi$ 下运行多个片段，并通过平均状态-动作对的回报来使用MC估算 $Q^\pi(s, a)$。
2. **策略改进**：更新策略 $\pi$，使其相对于估算的 $Q$ 值变为贪婪策略：对所有 $s$，$\pi(s) \leftarrow \underset{a}{\text{argmax}} Q(s, a)$。
3. 重复此过程，直到策略和价值函数稳定下来。

然而，这里有一个重要问题：试探。如果策略 $\pi$ 在早期就变得确定且贪婪（例如，它总是从状态 $s$ 选择动作 $a_1$），那么我们将只观察到对 $(s, a_1)$ 对的回报。我们将永远无法获取对状态 $s$ 中其他可能更好的动作（如 $a_2, a_3, \dots$）的经验（因此也无法更新 $Q$ 值估算）。智能体可能会陷入次优循环，因为它从未尝试其他选项。

为保证MC方法收敛到最优策略，我们需要保证所有状态-动作对都能持续被访问。这被称为保持试探。

"一种理论方式是假定*试探性开始*。这假定每个片段都从随机选择的状态-动作对 $(s, a)$ 开始，然后才遵循策略 $\pi$。这保证每个对都有非零的概率被选为起始点，从而保证最终都会被试探。虽然对理论证明有帮助，但试探性开始在实际使用中通常不切实际（例如，您可能无法随意设定机器人的初始状态和动作）。"

因此，实际的MC控制算法需要方法来保证持续试探，即使不依赖试探性开始。这通常会使用非完全确定性或非贪婪的策略，例如 $\epsilon$-贪婪策略，它们通常采取贪婪动作，但有时（以小概率 $\epsilon$）会选择一个随机动作。我们将在接下来的章节中讨论这些具体算法以及在线策略（on-policy）学习和离线策略（off-policy）学习之间的差异，这些差异直接关联到如何处理试探。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton, Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 5: Monte Carlo Methods
  强化学习领域的经典教材，全面介绍了蒙特卡洛预测和控制方法。
- [CS234: Reinforcement Learning | Lecture Notes: Monte Carlo Methods](http://web.stanford.edu/class/cs234/lectures/lecture4.pdf) — Emma Brunskill (2019)
  Publisher: Stanford University
  来自知名大学课程的讲义，涵盖蒙特卡洛预测和控制方法，包括Q值估计和探索问题。
- [Spinning Up in Deep RL: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/key_concepts.html) — Joshua Achiam and OpenAI (2018)
  Publisher: OpenAI
  来自领先人工智能研究机构的清晰教程，介绍了强化学习的核心概念，包括蒙特卡洛价值估计方法。
