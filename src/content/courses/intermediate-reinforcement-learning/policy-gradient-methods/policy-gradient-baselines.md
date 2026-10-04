---
course: "intermediate-reinforcement-learning"
chapter: "policy-gradient-methods"
lesson: "policy-gradient-baselines"
sourceId: 5147
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/policy-gradient-baselines"
title: "方差减少的基线"
description: "引入状态价值函数作为基线以减少方差。"
order: 6
plots: ["plots/5147-0.json"]
sourceHash: "c1031cd25b35aebef6ff3e4d0a2f173ab33e00de3f94175bd7f3873ecf8ac64c"
sourceCorrections: []
---

REINFORCE 算法使用完整的蒙特卡洛回报 $G_t$ 来调整梯度 $\nabla_\theta \log \pi(a_t|s_t; \theta)$ 的大小。尽管这提供了策略梯度的无偏估计，但 $G_t$ 的高方差常导致更新不稳定和收敛缓慢。设想一个所有奖励都很大且为正值的片段；此时所有时间步的 $G_t$ 都会很大，这会提高所执行所有动作的概率，即使在该成功的片段中有些动作远优于其他动作。反之，一个表现不佳的片段可能会过度惩罚所有执行的动作。

为减少这种情况，我们可以引入一个仅依赖于状态 $s_t$ 的**基线**函数 $b(s)$。在策略梯度更新中，我们从回报 $G_t$ 中减去这个基线：


$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi(a_{i,t}|s_{i,t}; \theta) (G_{i,t} - b(s_{i,t}))
$$


为什么这样做是有效的？非常重要的一点是，减去一个依赖于状态的基线并不会给梯度估计带来偏差。被减项的期望值为零：


$$
\mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi(a_t|s_t; \theta) b(s_t)] = \sum_{s} d^{\pi_\theta}(s) \sum_{a} \pi(a|s; \theta) [\nabla_\theta \log \pi(a|s; \theta) b(s)]
$$


$$
= \sum_{s} d^{\pi_\theta}(s) \sum_{a} \nabla_\theta \pi(a|s; \theta) b(s)
$$


$$
= \sum_{s} d^{\pi_\theta}(s) b(s) \nabla_\theta \sum_{a} \pi(a|s; \theta)
$$


$$
= \sum_{s} d^{\pi_\theta}(s) b(s) \nabla_\theta 1 = 0
$$


这里，$d^{\pi_\theta}(s)$ 是策略 $\pi_\theta$ 下的状态分布。由于被减项的期望为零，梯度的总期望保持不变。

尽管它不改变期望，但选择得当的基线能大幅降低梯度估计的方差。其思路是将回报围绕一个参照点进行居中处理。我们不再仅仅用总回报 $G_t$ 来调整梯度，而是根据该回报相对于我们通常从状态 $s_t$ 预期的表现 *好* 或 *坏* 的程度来调整。如果 $G_t > b(s_t)$，则动作 $a_t$ 带来了超出预期的结果，其概率应该增加。如果 $G_t < b(s_t)$，则动作带来了低于预期的结果，其概率应该降低（或增加得不那么强烈）。

### 基线的选择

怎样的基线算是好的？理想情况下，$b(s_t)$ 应该是从状态 $s_t$ 获得的预期回报的估计值。

1. **简单的常数基线：** 可以使用迄今为止在所有片段中遇到的所有回报的移动平均值。这种方法很简单，但不能适应特定状态的价值。
2. **状态价值函数 $V(s_t)$：** 更有效且常用的基线选择是状态价值函数 $V(s_t) = \mathbb{E}_{\pi_\theta}[G_t | S_t = s_t]$。这表示从状态 $s_t$ 开始并遵循当前策略 $\pi_\theta$ 时所能获得的预期回报。

使用 $V(s_t)$ 作为基线会得到项 $G_t - V(s_t)$。回想一下，动作价值函数是 $Q(s_t, a_t) = \mathbb{E}_{\pi_\theta}[G_t | S_t = s_t, A_t = a_t]$，优势函数定义为 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$。由于 $G_t$ 是从 $(s_t, a_t)$ 开始的回报的蒙特卡洛样本，因此 $G_t - V(s_t)$ 实际上是优势函数 $A(s_t, a_t)$ 的一个*估计*。

使用优势估计使缩放因子居中：带来优于该状态平均回报的动作获得正向增强，而带来差于平均回报的动作获得负向增强。这通常会使得方差大幅低于使用原始回报 $G_t$ 的情况。



![基线对梯度缩放因子方差的影响](plots/5147-0.json)



> 示例说明了在简化情况下，减去基线（此处 $V(s_t) \approx 10$）如何使缩放因子围绕零居中，并与使用原始回报 $G_t$ 相比减少其方差。

### 学习价值函数基线

当然，我们通常不知道当前策略下的真实 $V(s_t)$。因此，我们需要与策略同时学习它。这通常通过函数逼近来实现，常使用另一个神经网络 (neural network)。我们引入一个带参数 (parameter) $w$ 的价值函数逼近器 $V(s; w)$。

这个价值网络 $V(s; w)$ 经过训练，用于预测给定状态的预期回报。由于 REINFORCE 使用蒙特卡洛回报 $G_t$，一种常用的方法是通过最小化其预测值 $V(s_t; w)$ 与从收集到的轨迹中观察到的回报 $G_t$ 之间的均方误差（MSE）来训练价值网络：


$$
L(w) = \mathbb{E}_{\pi_\theta} [(G_t - V(s_t; w))^2]
$$


因此，在训练过程中，每次迭代（收集轨迹后）我们执行两个主要的更新：

1. **更新策略参数 $\theta$：** 使用带有学习基线的策略梯度：

   
   $$
   \Delta \theta = \alpha_\theta \nabla_\theta \log \pi(a_t|s_t; \theta) (G_t - V(s_t; w))
   $$
   

   注意，在计算策略梯度时，我们将 $V(s_t; w)$ 视为一个固定的基线值；我们不会将策略损失通过价值网络进行反向传播 (backpropagation)。
2. **更新价值函数参数 $w$：** 最小化预测值与实际回报之间的 MSE 损失：

   
   $$
   \Delta w = \alpha_w \nabla_w (G_t - V(s_t; w))^2 = \alpha_w \cdot 2 (G_t - V(s_t; w)) (-\nabla_w V(s_t; w))
   $$
   

这里，$\alpha_\theta$ 和 $\alpha_w$ 分别是策略和价值函数的学习率。

这种将学习到的价值函数 $V(s; w)$ 作为策略梯度方法基线的方法，构成了**Actor-Critic**算法的基础，我们将在下一章中详细讨论。策略网络 $\pi(a|s; \theta)$ 是“行动者”（Actor），它决定采取哪些动作，而价值网络 $V(s; w)$ 是“评判者”（Critic），它评估行动者访问的状态，并提供基线以减少行动者学习信号中的方差。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习的标准教科书。该版本详细介绍了策略梯度方法、使用基线进行方差缩减以及Actor-Critic算法的原理。第2版。
- [Actor-Critic Algorithms](https://doi.org/10.1137/S0363012901385691) — Vijay R. Konda and John N. Tsitsiklis (1999)
  Journal: Advances in Neural Information Processing Systems 12; Publisher: MIT Press; Pages: 1008-1014; DOI: [10.1137/S0363012901385691](https://doi.org/10.1137/S0363012901385691)
  本文介绍了Actor-Critic算法，这是一种使用基线来降低策略梯度方差的方法。
- [Lecture 9: Policy Gradients & Actor-Critic](https://stanford.edu/class/cs234/slides/CS234_Lecture9.pdf) — Emma Brunskill (2023)
  Journal: CS234: Reinforcement Learning; Publisher: Stanford University
  斯坦福大学CS234课程的讲义。这些讲义涵盖了策略梯度、基线对减小方差的益处，并介绍了Actor-Critic方法。
