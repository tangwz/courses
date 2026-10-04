---
course: "intermediate-reinforcement-learning"
chapter: "dqn-improvements-variants"
lesson: "dueling-dqn-architecture"
sourceId: 5123
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/dueling-dqn-architecture"
title: "对偶网络架构"
description: "将网络中的状态值和动作优势的估算分离。"
order: 3
plots: []
sourceHash: "6ebba935bdd86253f579ea5f74f0cb51a2148f77cc32b4eed990fc53b831553c"
sourceCorrections: []
---

提高深度Q网络（DQN）的性能可以涉及多种策略。双DQN等方法解决了Q值中的高估偏差，而另一种方法则侧重于修改神经网络 (neural network)本身的内在结构。考虑Q值，$Q(s, a)$。它隐含地包含两类信息：处于状态 $s$ 的总体好坏程度（状态值，$V(s)$），以及在该状态下执行动作 $a$ 比其他动作好多少（动作优势，$A(s, a)$）。

在许多情况中，状态 $V(s)$ 的值可能不强烈依赖于所采取的具体动作。例如，在驾驶场景中，如果前方遥远有一处不可避免的障碍物，当前状态的值（可能较低）主要由障碍物本身决定，无论你*现在*是轻微向左还是向右转。反之，当即刻行动很要紧时（例如躲避突然出现的坑洞），某个动作相对于其他动作的*优势*变得很要紧。

对偶网络架构明确承认这种分离。它没有一个直接计算所有动作 $Q(s, a)$ 的单一网络输出流，而是在一些初始共享层之后，将计算分为两条通路或“流”：

1. **价值流：** 输出一个标量值，这是对状态价值函数 $V(s)$ 的一个估算。
2. **优势流：** 输出一个维度等于动作数量的向量 (vector)，每个元素估算针对特定动作 $a$ 的动作优势函数 $A(s, a)$。

这两条流随后被结合起来，以生成每个动作的最终Q值估算。

### 结合价值与优势

回顾Q值、状态值和优势之间的关系：


$$
Q(s, a) = V(s) + A(s, a)
$$


这个等式说明，在状态 $s$ 中执行动作 $a$ 的价值，是处于状态 $s$ 的整体价值，再加上选择动作 $a$ 所带来的额外价值（或优势）。

对偶网络旨在分别学习 $V(s)$ 和 $A(s, a)$，然后将它们结合。然而，这里有一个需要注意的问题：仅凭贝尔曼更新中的目标Q值，我们无法唯一地恢复 $V(s)$ 和 $A(s, a)$。为什么？因为我们可以在 $V(s)$ 上加一个常数 $C$，并从所有 $A(s, a)$ 值中减去相同的常数 $C$，结果 $Q(s, a)$ 将保持不变：


$$
Q(s, a) = (V(s) + C) + (A(s, a) - C) = V(s) + A(s, a)
$$


这被称为*可识别性*问题。如果没有限制，网络可能会学习到，例如，一个非常高的 $V(s)$ 和相应较低（负值）的 $A(s, a)$ 值，反之亦然，这会使 $V$ 和 $A$ 的单独估算可能不稳定或难以解释。

### 稳定组合

为了解决可识别性问题并稳定学习，我们需要对优势流施加一个限制。一种常见且有效的方法是强制所有动作的平均优势为零。这意味着优势表示*偏离*状态值的部分。组合公式变为：


$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a') \right)
$$


这里，$|\mathcal{A}|$ 是可能动作的数量。通过减去平均优势，我们确保优势之和为零，有效地固定了 $V(s)$ 和 $A(s, a)$ 之间的尺度。这使得 $V(s)$ 成为对实际状态值更直接的估算，而 $A(s, a)$ 表示每个动作的相对偏好。

另一种有时使用的方法是将平均值替换为最大优势：


$$
Q(s, a) = V(s) + \left( A(s, a) - \max_{a' \in \mathcal{A}} A(s, a') \right)
$$


这会强制最佳动作的优势为零，所有其他动作都具有非正优势。均值减法方法通常更受青睐，因为它倾向于提高稳定性。

下图展示了对偶网络架构内的流程。

> 对偶网络的结构。输入状态特征通过共享层，然后分成单独的流，用于估算状态值 $V(s)$ 和动作优势 $A(s, a)$。这些之后通过一个特殊的聚合层结合起来，生成最终的Q值。

### 为何使用对偶架构？

分离价值和优势估算带来多种好处：

1. **学习效率提升：** 网络能更有效地学习状态值 $V(s)$，即使在该状态下许多动作的优势 $A(s, a)$ 很小或不相关。价值流的梯度直接更新 $V(s)$，而不会被所有动作输出所稀释。这在许多动作导致短期内相似的下一状态和回报的状态中，尤为有用。
2. **泛化能力更佳：** 通过独立于动作效果学习状态值，网络可能具有更强的泛化能力。它能更清楚地了解哪些状态本身是好的或坏的。
3. **侧重动作关联性：** 优势流学习在最需要时，专注于不同动作的相对重要性。

与双DQN类似，对偶网络架构常与DQN的其他改进结合使用，例如优先经验回放。这种组合，常被称为“对偶DDQN”，同时使用了减少高估偏差和更高效网络结构的好处，在许多基准任务上取得了顶尖的表现。

总之，对偶网络架构提供了一种更精巧的方式来组织Q值估算过程。通过将状态值 ($V(s)$) 与动作优势 ($A(s, a)$) 分离并仔细地重新组合它们，它使得网络能够更有效地学习，并且与标准DQN架构相比，通常能够带来更好的策略表现。

## 参考资料

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) — Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas (2016)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1511.06581](https://doi.org/10.48550/arXiv.1511.06581)
  介绍了Dueling网络架构，详细阐述了其结构、优势和稳定化技术。
- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 16.5
  提供了强化学习的全面概述，其中包括关于Dueling网络作为DQN改进的专门章节。
