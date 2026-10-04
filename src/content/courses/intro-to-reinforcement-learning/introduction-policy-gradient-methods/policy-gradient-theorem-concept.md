---
course: "intro-to-reinforcement-learning"
chapter: "introduction-policy-gradient-methods"
lesson: "policy-gradient-theorem-concept"
sourceId: 1666
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/policy-gradient-theorem-concept"
title: "策略梯度定理 (理念)"
description: "阐述策略梯度定理的主要思想，避免过多数学细节。"
order: 2
plots: []
sourceHash: "57efcb44b6db4e44fec269756b87a0538014ecfa77b625b9b6ced0607ad8a35c"
sourceCorrections: []
---

调整策略 $\pi_\theta(a|s)$ 的参数 (parameter) $\theta$ 的目标是使智能体表现更佳，具体来说，是最大化预期总回报。一个目标函数 $J(\theta)$ 表示这种预期回报，通常定义为自某个初始状态分布开始的预期回报。最终目的是找到能够最大化 $J(\theta)$ 的参数 $\theta$。

如何实现这种最大化呢？如果目标函数 $J(\theta)$ 简单，我们可以直接求解最大值。但在这里，$J(\theta)$ 复杂地依赖于策略与环境动态在潜在长序列的状态、动作和奖励中的相互影响。优化这类函数的常见方法是梯度上升。我们希望沿着使 $J(\theta)$ 增长最快的方向更新参数 $\theta$：


$$
\theta_{new} \leftarrow \theta_{old} + \alpha \nabla_\theta J(\theta)
$$


这里，$\alpha$ 是学习率，$\nabla_\theta J(\theta)$ 是目标函数相对于策略参数的梯度。难题在于计算这个梯度，即 $\nabla_\theta J(\theta)$。改变策略参数 $\theta$ 如何影响预期的未来总回报呢？考虑到这个回报取决于访问的状态和采取的动作序列，而这些又依赖于 $\theta$。

这正是**策略梯度定理**发挥作用的地方。它提供了一个重要的视角和一种实用的方法来计算或估计这个梯度，而无需了解环境的动态（例如状态转移概率）。该定理为梯度 $\nabla_\theta J(\theta)$ 建立了一个解析表达式，将其直接与策略 $\pi_\theta(a|s)$ 以及采取动作所关联的值联系起来。

尽管完整的推导涉及一些微积分，但其核心结果（一种常见形式）告诉我们，梯度与一个期望成比例：


$$
\nabla_\theta J(\theta) \propto E_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]
$$


让我们分解期望 $E_{\pi_\theta}[\cdot]$ 内部的组成部分，这个期望是在遵循当前策略 $\pi_\theta$ 产生的轨迹上计算的：

1. **$G_t$**: 这是在给定回合中从时间步 $t$ 开始的总回报（折扣奖励之和）。它表示在状态 $s$ 采取动作 $a$ 后在时间 $t$ 实际获得的值。
2. **$\nabla_\theta \log \pi_\theta(a|s)$**: 该项有时被称为“得分函数”。它可能看起来令人生畏，但其含义是：**参数 $\theta$ 的微小变化会如何影响在状态 $s$ 中采取动作 $a$ 的对数概率**。本质上，它指出了参数空间 ($\theta$) 中能增加从状态 $s$ 选择动作 $a$ 概率的方向。

因此，策略梯度定理将整体性能 $J(\theta)$ 的梯度与一个包含两项的期望联系起来：

- 我们有多大可能调整策略以在状态 $s$ 中采取动作 $a$ ($\nabla_\theta \log \pi_\theta(a|s)$)。
- 在状态 $s$ 中采取动作 $a$ 实际表现如何 ($G_t$)。

**基本原理：**

该定理告诉我们，为了增加预期回报，我们应该根据所采取的动作和获得的回报来调整策略参数 $\theta$。具体来说：

- 如果在状态 $s$ 采取动作 $a$ 导致了高回报 $G_t$，则项 $\nabla_\theta \log \pi_\theta(a|s)$ 会乘以一个大的正数。更新会将参数 $\theta$ 推向使特定动作 $a$ 在未来状态 $s$ 中*更可能*发生的方向。
- 如果在状态 $s$ 采取动作 $a$ 导致了低（或负）回报 $G_t$，则项 $\nabla_\theta \log \pi_\theta(a|s)$ 会乘以一个负数（或小的正数）。更新会将参数 $\theta$ 推向*相反*的方向，使动作 $a$ 在未来状态 $s$ 中*更不可能*发生。

期望 $E_{\pi_\theta}[\cdot]$ 意味着我们将这种效果平均到所有在遵循策略 $\pi_\theta$ 时遇到的状态-动作对上。平均而言，策略参数会调整，以增加导致良好结果的动作的概率，并降低导致不良结果的动作的概率。

**这为何重要？**

策略梯度定理意义重大，因为它以一种方式重新定义了梯度，我们可以利用智能体与环境交互收集的样本来*估计*它。我们不需要环境模型（即转移概率 $p(s', r | s, a)$）。我们只需运行策略，收集轨迹（状态、动作、奖励的序列），计算每一步的 $G_t$ 回报，计算 $\nabla_\theta \log \pi_\theta(a|s)$（因为我们定义了策略 $\pi_\theta$，所以可以做到），然后对乘积进行平均。

该定理构成了强化学习 (reinforcement learning)（RL）算法系列的一个理论基础，其中包括我们接下来将讨论的 REINFORCE 算法。这些算法通过采样经验并基于这种梯度估计原理应用更新来直接学习策略参数。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 13
  全面的教科书，涵盖强化学习的理论基础和算法，包括对策略梯度定理的详细解释和推导。第二版。
- [UCL Course on Reinforcement Learning - Lecture 7: Policy Gradient](https://www.scribd.com/document/267711204/Lecture-7-Policy-Gradient-David-Silver) — David Silver (2015)
  知名大学课程的讲义，对策略梯度方法和策略梯度定理提供了清晰直观的解释。
