# 不确定性下的乐观主义：UCB 方法

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies/ucb-methods-rl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然像 $\epsilon$-贪婪这样的简单试探方法能确保所有行动最终都会被尝试，但它们是随机进行试探，没有考虑到对每个行动的了解程度。偶然发现好策略在复杂问题中效率低下。我们需要更明智地引导试探，趋向有前景但尚不明晰的选择。“不确定性下的乐观主义”原则提供了一种正式的方法：根据现有信息，假设一切都尽可能地好。上限置信区间（UCB）算法体现了这一原则。

### 多臂老虎机类比

UCB 方法起源于多臂老虎机（MABs）的更简单环境。想象一下，你面对多台老虎机（“臂”），每台机器的回报都来自一个未知的概率分布。你的目标是在一系列拉动中最大化总回报。你应该继续拉动迄今为止给出最佳平均回报的臂（采纳当前最佳），还是尝试那些*可能*更好但尚未被充分试探的臂（进行试探）？

UCB 提供了一个在每个时间步 $t$ 选择臂 $a$ 的标准。它依赖于两个组成部分：

1. **估计价值 ($Q_t(a)$)：** 截至时间 $t$ 从臂 $a$ 获得的平均回报。这代表了当前的*采纳最佳*价值。
2. **不确定性奖励项：** 一个量化 (quantization)我们对臂 $a$ 真实价值不确定程度的项。对于被拉动次数较少的臂，此项会更高。

一种流行的 UCB 算法，UCB1，选择使这两个项之和最大的行动：


$$
A_t = \underset{a}{\mathrm{argmax}} \left[ Q_t(a) + c \sqrt{\frac{\ln N_t}{N_t(a)}} \right]
$$


此处：

- $A_t$ 是在时间 $t$ 选择的行动。
- $Q_t(a)$ 是在 $t$ 轮后行动 $a$ 的估计价值（平均回报）。
- $N_t$ 是在时间 $t$ 之前所有臂被拉动的总次数。
- $N_t(a)$ 是在时间 $t$ 之前行动 $a$ 被拉动的次数。
- $c$ 是一个常数（$c > 0$），控制了试探的程度。较高的 $c$ 值鼓励更多的试探。

项 $\sqrt{\frac{\ln N_t}{N_t(a)}}$ 充当不确定性度量。它随总游戏次数（$N_t$）对数增长，并随着特定臂被玩次数的增加（$N_t(a)$ 增加）而减小。这确保了那些不常尝试的行动，或估计值可能不准确的行动，会获得奖励，从而使它们更有可能被选中。如果 $N_t(a) = 0$，奖励项被认为是无限的，确保每个行动至少被尝试一次（假设进行了适当的初始化）。

这种“乐观”的方法有效地平衡了试探与采纳。具有高估计价值的行动受到偏爱，但不确定性高的行动也获得机会，尤其是在早期或其潜在上限具有竞争力时。

### 在强化学习 (reinforcement learning)中应用UCB

将 UCB 从无状态老虎机环境扩展到马尔可夫决策过程（MDPs）中的序贯决策，带来了挑战，但遵循相同的核心原则。状态增加了复杂性，因为一个行动的价值取决于它所处的状态。

调整 UCB 的常见方式是在每个状态 $s$ 内将其应用于行动选择。选择规则变为：


$$
A_t = \underset{a}{\mathrm{argmax}} \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]
$$


此处：

- $Q(s, a)$ 是在状态 $s$ 中采取行动 $a$ 的估计价值。
- $N(s)$ 是状态 $s$ 被访问的次数计数。
- $N(s, a)$ 是在状态 $s$ 中行动 $a$ 被采取的次数计数。

这需要维护 Q 值的估计（例如，使用 Q-learning 或 SARSA 更新），同时跟踪状态访问计数 $N(s)$ 和状态-行动计数 $N(s, a)$。



[交互图表：UCB 分数随时间变化的示意图 (c=1)](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies/ucb-methods-rl#plot-11cso3z)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        0.1,
        0.3,
        0.45,
        0.55,
        0.58,
        0.59,
        0.6
      ],
      "mode": "lines",
      "name": "Q(A)",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        1.41,
        0.81,
        0.6,
        0.46,
        0.4,
        0.36,
        0.33
      ],
      "mode": "lines",
      "name": "奖励项(A)",
      "line": {
        "color": "#a5d8ff",
        "dash": "dot"
      }
    },
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        1.51,
        1.11,
        1.05,
        1.01,
        0.98,
        0.95,
        0.93
      ],
      "mode": "lines+markers",
      "name": "UCB(A)",
      "line": {
        "color": "#4263eb",
        "width": 3
      }
    },
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        0.05,
        0.1,
        0.15,
        0.4,
        0.55,
        0.65,
        0.7
      ],
      "mode": "lines",
      "name": "Q(B)",
      "line": {
        "color": "#f76707"
      }
    },
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        1.41,
        1.15,
        0.95,
        0.6,
        0.51,
        0.44,
        0.4
      ],
      "mode": "lines",
      "name": "奖励项(B)",
      "line": {
        "color": "#ffd8a8",
        "dash": "dot"
      }
    },
    {
      "x": [
        1,
        5,
        10,
        20,
        30,
        40,
        50
      ],
      "y": [
        1.46,
        1.25,
        1.1,
        1.0,
        1.06,
        1.09,
        1.1
      ],
      "mode": "lines+markers",
      "name": "UCB(B)",
      "line": {
        "color": "#f76707",
        "width": 3
      }
    }
  ],
  "layout": {
    "title": {
      "text": "UCB 分数随时间变化的示意图 (c=1)"
    },
    "xaxis": {
      "title": "时间步 (示意)"
    },
    "yaxis": {
      "title": "价值 / 分数"
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": -0.3,
      "xanchor": "center",
      "x": 0.5
    }
  }
}
```

</details>



> Q 值、不确定性奖励项以及由此产生的 UCB 分数的演变，针对真实平均值分别为 0.6 和 0.7 的两个行动（A 和 B）。随着行动 B 更高潜力的得以展现，其 UCB 分数最终超过了 A。

### 挑战与调整

上述直接应用在表格化设置或具有离散、可管理状态空间的环境中效果良好。然而，在具有大型或连续状态和行动空间的环境中，它面临重大障碍：

1. **计数跟踪：** 当状态连续或状态空间庞大时，维护精确计数 $N(s)$ 和 $N(s, a)$ 变得不可能。需要函数逼近技术来估计这些计数或密度模型。从密度模型或哈希函数导出的伪计数通常用作近似值（本章稍后讨论）。
2. **函数逼近：** 当使用深度神经网络 (neural network)或其他函数逼近器来估计 $Q(s, a)$ 时，函数逼近器与 UCB 奖励项之间的相互影响需要仔细考量。简单地将奖励项添加到近似 Q 值可能在统计上不合理或不稳定。
3. **试探常数：** 参数 (parameter) $c$ 平衡了试探与采纳。其最优值可能取决于具体问题，并可能需要仔细调整。从老虎机问题导出的理论值可能不直接适用。
4. **非平稳性：** 标准 UCB 假设奖励分布是平稳的。在强化学习 (reinforcement learning)中，状态-行动对的有效奖励分布可能发生变化，因为策略会演变，并到达不同的后续状态。

尽管存在这些挑战，UCB 的核心原则影响了许多高级试探方法。例如，在基于模型的强化学习中，UCB 经常用于蒙特卡洛树搜索（MCTS）等规划算法中，以引导模拟过程趋向不确定但可能有益的行动序列（如第 5 章所示）。结合函数逼近和处理广阔状态空间的改进也得到了发展，通常将 UCB 的思想与其他方法（如自举法或内在激励）结合。

### 总结

UCB 方法通过显式建模不确定性，提供了一种原则性的方式来管理试探与采纳的权衡。通过向估计的行动价值添加不确定性奖励项，UCB 鼓励对那些尚不清晰但可能带来高回报的行动进行试探。尽管在大规模强化学习 (reinforcement learning)中的直接应用需要进行调整以处理广阔的状态空间和函数逼近，但“不确定性下的乐观主义”这一思想在设计复杂序贯决策问题的试探策略时仍具有重要意义。这代表了迈向更具方向性且可能更高效地发现最优策略的一步。

## 参考资料

- [Finite-time Analysis of the Multiarmed Bandit Problem](https://doi.org/10.1023/A:1013689704352) — Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer (2002)
  Journal: Machine Learning; Publisher: Springer Nature B.V.; Volume: 47; Pages: 235-256; DOI: [10.1023/A:1013689704352](https://doi.org/10.1023/A%3A1013689704352)
  提供了多臂赌博机问题中UCB1算法的原始分析和算法。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  标准教材，涵盖强化学习基础概念，包括多臂赌博机和UCB方法。
- [Bandit Based Monte-Carlo Planning](https://doi.org/10.1007/11871842_29) — Levente Kocsis, Csaba Szepesvári (2006)
  Journal: Machine Learning: ECML 2006 (ECML 2006); Publisher: Springer-Verlag Berlin Heidelberg; Pages: 282-293; DOI: [10.1007/11871842_29](https://doi.org/10.1007/11871842_29)
  介绍了树上置信上限（UCT），将UCB应用于蒙特卡洛树搜索以进行高效搜索。

---

[上一节](01-%E5%86%8D%E8%B0%88%E8%AF%95%E9%94%99-%E5%88%A9%E7%94%A8%E6%9D%83%E8%A1%A1.md) · [下一节](03-%E6%A6%82%E7%8E%87%E5%8C%B9%E9%85%8D%EF%BC%9A%E6%B1%A4%E6%99%AE%E6%A3%AE%E9%87%87%E6%A0%B7.md)
