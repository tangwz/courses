---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "variance-reduction-baselines"
sourceId: 3386
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/variance-reduction-baselines"
title: "降低方差的基线"
description: "使用状态值函数作为基线，降低策略梯度估计中的方差。"
order: 3
plots: ["plots/3386-0.json"]
sourceHash: "9baace5efbd724e27714d09658cd2cd186398c96ffd1635d606f727d42e5a350"
sourceCorrections: []
---

REINFORCE算法虽然是策略梯度的根本，但在实践中常因其梯度估计固有的高方差而遇到困难。策略更新依赖于在状态 $s_t$ 中执行动作 $a_t$ 后观察到的总回报 $G_t$。由于 $G_t$ 会根据后续的随机转移和动作而变化很大，所以产生的梯度估计 $\nabla_\theta \log \pi_\theta(a_t|s_t) G_t$ 可能噪声较大。这种噪声会减慢学习速度，并可能阻止策略稳定收敛到一个好的解决方案。想象一下，你试图根据剧烈波动的反馈来调整一个旋钮；很难知道哪个方向确实更好。

Actor-Critic方法通过在策略梯度计算中引入一个*基线*，提供了一个有力的解决方案。其核心思想出乎意料地简单：从回报 $G_t$ 中减去一个仅依赖于状态 $s_t$ 的值，我们称之为 $b(s_t)$。修改后的策略梯度更新项变为：


$$
\nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))
$$


为什么这有效呢？ 重要的是，减去一个依赖于状态的基线并*不*会改变策略梯度的期望值，这意味着它不会在更新方向上引入偏差。我们可以通过数学方式证明这一点。我们减去的项的期望值为：


$$
\mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t) \right] = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) b(s)
$$


使用恒等式 $\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$，这变为：


$$
\sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) b(s) = \sum_s d^{\pi_\theta}(s) b(s) \nabla_\theta \sum_a \pi_\theta(a|s)
$$


由于对于任何状态 $s$，$\sum_a \pi_\theta(a|s) = 1$，所以它相对于 $\theta$ 的梯度为零：$\nabla_\theta 1 = 0$。因此，被减项的期望值为零：


$$
\mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t) \right] = 0
$$


这证实了减去一个依赖于状态的基线 $b(s_t)$ 能够保持策略梯度估计的无偏性。

### 基线如何降低方差

虽然期望梯度保持不变，但梯度估计的方差可以大幅降低。考虑项 $(G_t - b(s_t))$。如果我们选择 $b(s_t)$ 作为从状态 $s_t$ 获得的*平均*回报的一个良好估计，那么这一项表示实际观察到的回报 $G_t$ 与该状态的平均预期相比是*更好*还是*更差*。

- 如果 $G_t > b(s_t)$：动作 $a_t$ 导致了优于平均水平的结果。该项为正，强化动作 $a_t$。
- 如果 $G_t < b(s_t)$：动作 $a_t$ 导致了劣于平均水平的结果。该项为负，抑制动作 $a_t$。

通过将回报值围绕状态特定的平均值进行中心化，更新的幅度被缩小。更新不再基于高绝对回报（这可能仅仅源于一个通常有回报的状态）产生大的正向更新，而是侧重于动作在*该特定情境下*的相对质量。这导致了更稳定且通常更快的学习过程。



![简单基线对回报的影响](plots/3386-0.json)



> 轨迹中的样本回报 ($G_t$)。减去一个基线（此处为总体平均回报）将用于策略更新的值中心化到零附近。正值对应于优于基线的回报，负值对应于劣于基线的回报。这种中心化有助于降低更新方差。

### 状态值函数作为最佳基线

$b(s_t)$ 的最佳选择是什么？虽然一个简单的常数基线（比如一个回合的平均回报）能有所帮助，但一个效果好得多的基线是状态值函数 $V^{\pi_\theta}(s_t)$。这个函数根据定义表示从状态 $s_t$ 开始并遵循当前策略 $\pi_\theta$ 所能获得的期望回报。

使用 $V(s_t)$ 作为基线，更新项变为：


$$
\nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - V(s_t))
$$


项 $(G_t - V(s_t))$ 是**优势函数** $A^{\pi_\theta}(s_t, a_t)$ 的一个估计。优势函数衡量了在状态 $s_t$ 中执行动作 $a_t$ 比策略 $\pi_\theta$ 从状态 $s_t$ 平均选择的动作好多少。形式上：


$$
A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)
$$


其中 $Q^{\pi_\theta}(s_t, a_t)$ 是动作值函数。由于 $G_t$ 是 $Q^{\pi_\theta}(s_t, a_t)$ 的蒙特卡洛样本估计，项 $(G_t - V(s_t))$ 作为优势 $A^{\pi_\theta}(s_t, a_t)$ 的样本估计。

理论上，使用状态值函数 $V(s_t)$ 作为基线是最佳选择（在最小化梯度估计方差的意义上），在所有仅依赖于状态 $s_t$ 的函数中。

### 评价者（Critic）的出现

这自然地引向了Actor-Critic架构。我们需要一种方法来估计 $V(s_t)$ 以将其用作基线。这正是*评价者*（critic）的角色。

- **执行者**（Actor）负责学习和更新策略参数 (parameter) $\theta$，通常使用带优势估计的梯度上升：
  
  $$
  \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t, a_t)
  $$
  
  （其中 $A(s_t, a_t)$ 被估计，例如通过 $G_t - V(s_t)$ 或我们将在后面看到的其他方法）。
- **评价者**（Critic）负责学习状态值函数 $V(s_t; \phi)$（有时是动作值函数 $Q(s_t, a_t; \phi)$）的估计，该函数由 $\phi$ 参数化。它从执行者所经历的转移中学习，通常使用时序差分（TD）学习方法，例如TD(0)。评价者的输出 $V(s_t; \phi)$ 随后被执行者用作基线 $b(s_t)$。

> 使用 $V(s)$ 作为基线的Actor-Critic架构图。执行者根据策略选择动作。评价者通过学习 $V(s)$ 来评估状态。评价者的值估计 $V(s)$ 被用作基线来计算优势估计，这反过来为更新执行者的策略提供了较低方差的信号。这两个组件都从与环境的交互中学习。

总而言之，引入一个依赖于状态的基线，特别是状态值函数 $V(s_t)$，是一种用于降低策略梯度估计方差而不引入偏差的强大技术。这自然地促使了Actor-Critic框架的形成，其中评价者学习值函数以提供此基线，而执行者则使用产生的优势信号更新策略。后续章节将讨论A2C/A3C等有效实现此思想的特定算法，并引入如广义优势估计（GAE）等进一步改进。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapters 13-14
  一本内容全面的教科书，详细介绍了策略梯度方法、基线在方差缩减中的作用以及Actor-Critic架构的原理。
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/file/464d7c81ceb01d2fe630f146a06655c6-Paper.pdf) — Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour (1999)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 12; Pages: 1057-1063; DOI: [10.5555/3008751.3008851](https://doi.org/10.5555/3008751.3008851)
  引入了策略梯度定理，并正式讨论了减去依赖于状态的基线（如价值函数）如何在不引入偏差的情况下减少策略梯度估计的方差。
- [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html) — Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: Proceedings of The 33rd International Conference on Machine Learning; Publisher: PMLR; Volume: 48; Pages: 1928-1937
  一篇里程碑式的论文，展示了通过Actor-Critic架构成功进行深度强化学习。该方法利用价值函数作为基线，通过异步更新来稳定训练并减少方差。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  引入了广义优势估计（GAE），一种在优势函数估计中实现良好偏差-方差权衡的方法，这对于先进的策略梯度方法很重要。
