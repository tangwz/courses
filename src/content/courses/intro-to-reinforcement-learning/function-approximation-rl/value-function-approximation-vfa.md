---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "value-function-approximation-vfa"
sourceId: 1651
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/value-function-approximation-vfa"
title: "价值函数近似 (VFA)"
description: "介绍使用参数化函数近似价值函数的思想。"
order: 2
plots: []
sourceHash: "b03a0db757b62acfca7b126dc4b421e2927f2894f5630838ea2e75b859bdb9c6"
sourceCorrections: []
---

在处理大型或连续状态空间时，将每个状态或状态-动作对的值存储在表格中变得不可行。想象一下，为自动驾驶汽车创建一个Q表，其状态包括摄像头图像和激光雷达数据等传感器读数，状态空间几乎是无限的！表格方法根本无法扩展。

解决方法是从显式存储转向估计。我们不再学习每个状态或状态-动作对的精确值 $v_\pi(s)$ 或 $q_\pi(s, a)$，而是学习一个参数 (parameter)化函数来*近似*这些值。这种技术被称为**价值函数近似 (VFA)**。

我们引入一个函数，称之为 $\hat{v}$，它以状态 $s$ 和参数向量 (vector) $\mathbf{w}$ 作为输入，并输出一个估计的状态值：


$$
\hat{v}(s, \mathbf{w}) \approx v_\pi(s)
$$


类似地，我们可以用函数 $\hat{q}$ 来近似动作价值函数 $q_\pi(s, a)$，它以状态 $s$、动作 $a$ 和参数 $\mathbf{w}$ 作为输入：


$$
\hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a)
$$


这里，$\mathbf{w}$ 是一个权重 (weight)或参数向量（例如，线性模型中的系数或神经网络 (neural network)中的权重）。主要思路是 $\mathbf{w}$ 中的参数数量远小于状态总数 $|S|$ 或状态-动作对总数 $|S| \times |A|$。例如，我们可能有数百万个状态，但 $\mathbf{w}$ 中可能只需要几百或几千个参数。

### 泛化的能力

使用函数近似的主要优点是**泛化能力**。因为函数近似器根据参数 (parameter) $\mathbf{w}$ 学习一种关系，它甚至可以估计以前从未遇到过或很少遇到的状态的值。如果两个状态 $s_1$ 和 $s_2$ 被认为是“相似的”（这通常由我们如何表示它们来决定，我们很快会讨论），函数近似器很可能会产生相似的值估计 $\hat{v}(s_1, \mathbf{w})$ 和 $\hat{v}(s_2, \mathbf{w})$。这使得智能体可以借助在状态空间某部分获得的经验，在其他相似部分做出更好的决策。相比之下，表格方法独立对待每个状态；了解 $s_1$ 对 $s_2$ 没有帮助。

### 函数近似器的类型

我们可以为 $\hat{v}$ 和 $\hat{q}$ 使用各种类型的函数。常见选择包括：

1. **线性函数：** 它们将值近似为表示状态（或状态-动作对）的特征的线性组合。它们简单且易于理解。
2. **神经网络 (neural network)：** 它们是强大的非线性函数近似器，能够学习状态/动作与其值之间的复杂关系。深度神经网络构成了深度强化学习 (reinforcement learning)的根基。
3. 其他方法，如决策树、瓦片编码、径向基函数等，也可以使用。

在本课程中，我们将主要关注线性方法，并介绍使用神经网络的原理。

### 学习目标

我们使用VFA的目标是找到参数 (parameter)向量 (vector) $\mathbf{w}$，使我们的近似值 $\hat{v}(s, \mathbf{w})$ 或 $\hat{q}(s, a, \mathbf{w})$ 尽可能接近真实价值函数 $v_\pi(s)$ 或 $q_\pi(s, a)$（或最优的 $v_*(s)$ 或 $q_*(s, a)$）。这通常被表述为最小化一个误差目标，例如在遇到的状态分布上的均方价值误差（MSVE）：


$$
MSVE(\mathbf{w}) = \sum_{s \in S} d(s) [v_\pi(s) - \hat{v}(s, \mathbf{w})]^2
$$


其中 $d(s)$ 是一个权重 (weight)，表示我们对状态 $s$ 中误差的重视程度。

虽然直接最小化MSVE通常是目标，但我们使用的算法（如TD学习的变体）由于RL更新的性质，实际上优化的是略微不同的目标。

### 与监督学习 (supervised learning)的联系

你可能会发现，寻找参数 (parameter) $\mathbf{w}$ 听起来与监督学习相似。我们有输入（状态 $s$，或状态-动作对 $(s,a)$），并且我们想预测目标输出（真实值 $v_\pi(s)$ 或 $q_\pi(s,a)$）。确实，我们将使用像梯度下降 (gradient descent)这样的技术来更新 $\mathbf{w}$，这在监督学习中很常见。

然而，存在一个重要的不同：在强化学习 (reinforcement learning)中，我们通常不知道*真实*的目标值 $v_\pi(s)$ 或 $q_\pi(s,a)$。相反，我们使用从与环境交互中获得的这些值的*估计*（例如，观察到的奖励和随后的估计值，如在TD学习中）。这意味着我们的目标值通常是嘈杂的、有偏的，并且非平稳的（它们随着策略和价值估计的改进而变化），这与标准监督学习相比带来了独特的难题。

在接下来的部分中，我们将了解如何使用特征表示状态，以及如何在强化学习框架内应用基于梯度的方法来学习线性函数近似器和非线性函数近似器的参数 $\mathbf{w}$。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本基础教材，涵盖了价值函数近似，包括线性方法和理论基础。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  一篇开创性论文，展示了深度神经网络在近似Q值方面的应用，从而在复杂环境中实现高性能控制。
