---
course: "intro-to-reinforcement-learning"
chapter: "introduction-policy-gradient-methods"
lesson: "learning-policies-directly"
sourceId: 1665
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/learning-policies-directly"
title: "直接学习策略"
description: "介绍直接对策略进行参数化和优化的思想，而不是价值函数。"
order: 1
plots: []
sourceHash: "5f597936e8ed3146b482f341e316e01be1a134c9dca7504a4e85bb5b611f5649"
sourceCorrections: []
---

强化学习 (reinforcement learning)中的方法通常以学习处于特定状态或在状态中采取特定行动的*价值*为中心。Q学习和SARSA等算法首先估计价值函数（如$Q(s, a)$），然后根据这些估计值来确定最佳行动，通常是通过选择估计值最高的行动。这种方法虽然效果好，但面临一些问题，尤其是在处理连续动作空间或当最优行为本身固有的随机性（随机策略）时。

策略梯度方法提供另一种视角。我们不学习价值函数，而是直接学习策略本身。我们将策略定义为一个参数 (parameter)化的函数，我们将其记作$\pi_\theta(a|s)$，它输出在状态$s$中采取行动$a$的概率。此函数的参数由$\theta$表示。例如，$\theta$可以是神经网络 (neural network)的权重 (weight)和偏置 (bias)。

主要思想是直接调整参数$\theta$，以提升策略的质量。但“提升”意味着什么？在强化学习中，我们的目标通常是最大化期望的总折现回报。我们可以定义一个性能指标，通常表示为$J(\theta)$，它代表通过遵循策略$\pi_\theta$所获得的期望回报。对于幕式任务，这通常是起始状态的价值：


$$
J(\theta) = V^{\pi_\theta}(s_0) = E_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} | S_0 = s_0 \right]
$$


我们的目标是找到使这个性能指标$J(\theta)$最大化的参数$\theta$。

### 为何直接学习策略？

直接对策略进行参数 (parameter)化带来几个好处：

1. **连续动作空间：** 像Q学习这样的基于价值的方法难以处理连续动作。如果动作空间是连续的，则在每一步都需要一个优化过程来找到使$Q(s, a)$最大化的动作$a$。策略梯度方法可以更自然地处理这个问题。参数化的策略$\pi_\theta(a|s)$可以设计为直接输出动作的概率分布的参数（例如，针对连续动作的高斯分布的均值和标准差）或连续动作本身。
2. **随机策略：** 在某些情况下，最优策略是随机的。考虑一个像剪刀石头布这样的游戏，对手可以利用确定性策略。随机（随机）策略是最佳的。基于价值的方法通常隐含地学习确定性策略（例如，通过总是选择具有最大Q值的动作）。策略梯度方法可以通过输出概率$\pi_\theta(a|s)$来明确地学习随机策略。
3. **更简单的表示：** 有时，策略函数比价值函数更容易近似。直接参数化可能需要更少的参数或更简单的函数近似器，与表示价值函数可能复杂的形状相比。

### 通过梯度上升进行优化

既然我们有一个参数 (parameter)化函数$\pi_\theta$和一个我们想要最大化的目标函数$J(\theta)$，这看起来像一个优化问题。我们可以使用与监督学习 (supervised learning)中相似的技术，但不是梯度*下降*（为了最小化损失），我们使用梯度*上升*（为了最大化性能）。

策略参数$\theta$的基本更新规则是：


$$
\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)
$$


这里：

- $\theta_k$表示在第$k$次迭代时的策略参数。
- $\alpha$是一个正的步长参数，通常称为学习率。
- $\nabla_\theta J(\theta_k)$是性能指标关于策略参数$\theta$的梯度。这个梯度向量 (vector)指向$J(\theta)$最大增长的方向。

通过反复计算梯度并沿着该方向迈步，我们迭代地优化策略参数$\theta$，以获得更高的期望回报。

### 策略的参数 (parameter)化

我们如何具体表示$\pi_\theta(a|s)$？选择取决于动作空间的性质。

- **离散动作：** 一种常见的方法是使用函数近似器（如神经网络 (neural network)），它以状态$s$作为输入，并为每个可能的动作$a$输出一个分数或偏好$h(s, a, \theta)$。然后可以使用softmax函数将其转换为概率：

  
  $$
  \pi_\theta(a|s) = \frac{e^{h(s, a, \theta)}}{\sum_{a'} e^{h(s, a', \theta)}}
  $$
  

  参数$\theta$是函数近似器的参数（例如，网络权重 (weight)）。
- **连续动作：** 一种流行的方法是让函数近似器输出概率分布的参数。例如，对于单个连续动作，网络可能会输出高斯分布的均值$\mu_\theta(s)$和标准差$\sigma_\theta(s)$。然后从这个分布中采样动作：

  
  $$
  a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)
  $$
  

  策略概率密度然后由高斯概率密度函数（PDF）给出。参数$\theta$再次是决定$\mu$和$\sigma$的网络参数。

主要问题以及本章大部分内容的重点在于估计梯度$\nabla_\theta J(\theta)$。与我们有明确标签的监督学习 (supervised learning)不同，在强化学习 (reinforcement learning)中，一个动作的“正确性”只通过在可能很长的轨迹中接收到的后续奖励体现出来。策略梯度定理（我们接下来将介绍）提供一个理论依据，用于计算或估计这个重要梯度。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapter 13
  强化学习的标准教材，其中有一个详细章节介绍策略梯度方法，涵盖核心概念和策略梯度定理。
- [Foundations of Deep Reinforcement Learning: The Actor-Critic Perspective](https://mitpress.mit.edu/books/foundations-deep-reinforcement-learning) — Marlos C. Machado, Marc G. Bellemare, and Michael Bowling (2023)
  Publisher: MIT Press
  对深度强化学习进行了当代且严谨的处理，为策略梯度和Actor-Critic方法提供了进阶观点。
