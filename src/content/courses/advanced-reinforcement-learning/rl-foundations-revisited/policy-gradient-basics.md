---
course: "advanced-reinforcement-learning"
chapter: "rl-foundations-revisited"
lesson: "policy-gradient-basics"
sourceId: 3354
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited/policy-gradient-basics"
title: "策略梯度方法入门"
description: "回顾策略梯度定理和REINFORCE算法，作为高级方法的基础。"
order: 5
plots: []
sourceHash: "2ba6b0a012ca0f5f84ca745914cf1fa01cda56f27a67edd45ce6f7617b4384d3"
sourceCorrections: []
---

基于价值的方法，如Q学习，其主要思想是学习状态-行动对的价值 $Q(s, a)$，然后根据这些价值得到策略（通常是隐式地，如epsilon-贪婪）。策略梯度方法提供了一种根本不同的方法：它们直接对策略进行参数 (parameter)化，并优化其参数以使预期回报最大化。

我们将策略记作 $\pi_\theta(a|s)$，表示在策略参数 $\theta$ 下，在状态 $s$ 中采取行动 $a$ 的概率。目标是找到使性能目标函数最大化的参数 $\theta$，通常是从初始状态分布开始的预期总折扣奖励。我们可以将此目标定义为：


$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right]
$$


这里，$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)$ 表示通过遵循策略 $\pi_\theta$ 生成的轨迹，$\gamma$ 是折扣因子，而 $r(s_t, a_t)$ 是在状态 $s_t$ 采取行动 $a_t$ 后获得的奖励。

主要挑战是计算此目标函数关于策略参数 $\nabla_\theta J(\theta)$ 的梯度。直接对目标函数求导似乎很困难，因为期望取决于由 $\pi_\theta$ 引起的轨迹分布，而该分布本身又依赖于 $\theta$。策略梯度定理提供了一种便捷的解决办法。它指出目标函数的梯度可以表示为：


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) Q^{\pi_\theta}(s_t, a_t) \right]
$$


或者，使用 $\pi_\theta$ 下访问的状态和行动的期望表示法：


$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta(a|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right]
$$


其中 $d^{\pi_\theta}(s)$ 是策略 $\pi_\theta$ 下的平稳状态分布。术语 $\nabla_\theta \log \pi_\theta(a|s)$ 有时被称为“分数函数”。它告诉我们如何调整参数 $\theta$，以提高在状态 $s$ 中采取行动 $a$ 的对数概率。该定理告诉我们，要用行动价值函数 $Q^{\pi_\theta}(s, a)$ 来加权这个方向。直观上，这讲得通：我们希望增加导致高于平均回报的行动的概率，并降低导致低于平均回报的行动的概率。

### REINFORCE算法

策略梯度定理提供了梯度的形式，但我们仍然需要 $Q^{\pi_\theta}(s_t, a_t)$，这通常是未知的。REINFORCE算法（也称为蒙特卡洛策略梯度）提供了一种使用采样轨迹来估计此梯度的简单方法。

REINFORCE使用从时间步 $t$ 开始的完整回报 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$，作为 $Q^{\pi_\theta}(s_t, a_t)$ 的无偏样本估计。由于我们正在采样轨迹，我们可以用采样轨迹的平均值来近似期望。对于单个轨迹，每个时间步 $t$ 的更新方向是 $G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$。

基本的REINFORCE算法流程如下：

1. 初始化策略参数 (parameter) $\theta$。
2. 重复：
   a. 通过遵循当前策略 $\pi_\theta$ 生成一个完整的片段轨迹 $\tau = (s_0, a_0, r_1, \dots, s_T, a_T, r_{T+1})$。
   b. 对于每个时间步 $t = 0, 1, \dots, T$：
   i. 计算从该时间步起的回报：$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$。
   ii. 计算此步骤的策略梯度估计：$g_t = G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$。
   c. 使用片段中累积的梯度更新策略参数（或者通常通过对片段中的 $g_t$ 求和并进行一步更新）：$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} g_t$，其中 $\alpha$ 是学习率。

这是REINFORCE更新逻辑的可视化表示：

> REINFORCE算法的示意性概述：使用当前策略 $\pi_\theta$ 生成轨迹，计算每个步骤的回报 $G_t$ 和分数函数 $\nabla \log \pi_\theta(a_t|s_t)$，然后更新策略参数 $\theta$。

### 优点与缺点

像REINFORCE这样的策略梯度方法有几个优点：

1. **直接优化：** 它们直接优化我们关注的量：预期回报。
2. **连续行动：** 它们自然地处理连续行动空间，在其中找到行动的最大值（如Q学习的更新 $\max_{a'} Q(s', a')$ 所需）是有问题的。我们可以使用高斯分布等来定义 $\pi_\theta(a|s)$，其参数 (parameter)是神经网络 (neural network)的输出。
3. **随机策略：** 它们可以学习随机策略，这在部分可观察的环境中或面对对手时可能有利。基于价值的方法通常学习确定性策略（从Q值的贪婪选择中得出）。

然而，REINFORCE也有一个显著的缺点：

1. **高方差：** $Q^{\pi_\theta}(s_t, a_t)$ 的蒙特卡洛估计 $G_t$ 可能具有非常高的方差。单个轨迹可能纯粹由于偶然性而异常好或坏，导致梯度估计噪声大，学习缓慢且不稳定。

这种高方差是开发更先进策略梯度方法的主要原因，例如行动者-评论者算法（将在第3章讨论），这些方法使用学习到的价值函数（评论者）来减小策略梯度估计（来自行动者）的方差。

### 使用函数逼近的策略表示

正如价值函数可以用于大型状态空间，使用线性函数或神经网络 (neural network)进行逼近一样，策略 $\pi_\theta(a|s)$ 也可以使用函数逼近器来表示。在深度强化学习 (reinforcement learning)中，$\pi_\theta$ 通常由一个神经网络表示，该网络接收状态 $s$ 作为输入，并输出行动分布的参数 (parameter)（例如，离散行动的概率，或连续行动的均值和标准差）。参数 $\theta$ 然后对应于该神经网络的权重 (weight)和偏差。这种与函数逼近的结合使得策略梯度方法能够扩展到高维状态和行动空间的复杂问题，构成了本课程后续讨论的许多高级技术的基础。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  本书是强化学习的经典参考文献，提供了策略梯度方法（包括策略梯度定理和REINFORCE算法）的全面解释。
- [Lecture 6: Policy Gradient](https://www.davidsilver.uk/wp-content/uploads/2020/02/PolicyGradient.pdf) — David Silver (2015)
  Journal: UCL Course on Reinforcement Learning; Publisher: University College London; Pages: 66
  大卫·西尔弗的系列讲座影响深远。第六讲专门介绍了策略梯度方法、策略梯度定理和REINFORCE算法，常从与深度强化学习相关的现代视角进行阐述。
- [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: Proceedings of The 33rd International Conference on Machine Learning; Publisher: PMLR; Volume: 48; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  这篇论文虽然介绍了A3C算法，但其关键部分在于讨论了用于策略梯度方差缩减的优势函数，这与REINFORCE算法提及的缺点直接相关。
