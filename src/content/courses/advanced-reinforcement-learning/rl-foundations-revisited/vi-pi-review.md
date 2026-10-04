---
course: "advanced-reinforcement-learning"
chapter: "rl-foundations-revisited"
lesson: "vi-pi-review"
sourceId: 3349
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited/vi-pi-review"
title: "价值迭代与策略迭代"
description: "概述经典的动态规划方法：用于求解MDP的价值迭代和策略迭代。"
order: 3
plots: ["plots/3349-0.json"]
sourceHash: "dd5805a03d5a8eb0feff6daf5aebb42a4a95077efdeb651c9d3fd1d7d533dda8"
sourceCorrections: []
---

马尔可夫决策过程（MDP）的重要组成部分和贝尔曼方程是强化学习 (reinforcement learning)中的必要内容。用于求解当模型（转移和奖励）已知时的MDP，存在两种基本动态规划算法：策略迭代和价值迭代。这些方法为许多在模型未知情况下运行的强化学习技术提供了根本依据。

### 策略迭代（PI）

策略迭代是一种通过重复执行两个步骤来寻找最优策略$\pi^*$的方法：评估当前策略，然后根据评估结果改进策略。它从任意初始策略$\pi_0$开始，并在这两个阶段之间交替进行，直到策略不再发生变化。

#### 1. 策略评估

在此步骤中，我们计算当前策略$\pi$的状态价值函数$V^\pi(s)$。回想一下$V^\pi$的贝尔曼期望方程：


$$
V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]
$$


对于一个具有$|S|$个状态的有限MDP，这个方程定义了一个包含$|S|$个未知数（所有$s \in S$的$V^\pi(s)$）的$|S|$个线性方程组。这个系统可以直接求解。然而，一种常用的迭代方法是重复将贝尔曼期望备份算子应用于初始估计值$V_k(s)$，直到这些值收敛：


$$
V_{k+1}(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_k(s')]
$$


对所有状态$s$执行此迭代更新，直到价值函数在迭代之间的变化足够小（例如，$\max_s |V_{k+1}(s) - V_k(s)| < \theta$）。结果是对$V^\pi$的精确估计。

#### 2. 策略改进

一旦我们有了当前策略$\pi$的价值函数$V^\pi$，我们就会检查是否可以对其进行改进。我们通过考虑对于每个状态$s$，是否采取与$\pi$规定不同的行动会产生更好的预期回报来做到这一点。这涉及使用计算出的$V^\pi$来计算行动价值函数$Q^\pi(s, a)$：


$$
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]
$$


然后，我们定义一个新的、可能已改进的策略$\pi'$，它相对于$Q^\pi(s, a)$采取贪婪行动：


$$
\pi'(s) = \arg\max_a Q^\pi(s, a)
$$


这意味着$\pi'(s)$选择能够最大化预期一步展望值的行动，假设之后我们遵循原始策略$\pi$（如$V^\pi$所捕获）。策略改进定理保证这个新策略$\pi'$至少与$\pi$一样好，甚至可能更好。也就是说，对于所有$s \in S$，都有$V^{\pi'}(s) \ge V^\pi(s)$。

#### 策略迭代算法

整个算法在这两个步骤之间交替进行：

1. 任意初始化策略$\pi_0$。
2. **重复**：
   a. **策略评估**：计算$V^{\pi_k}$（例如，使用迭代贝尔曼期望备份）。
   b. **策略改进**：使用$V^{\pi_k}$计算$Q^{\pi_k}(s, a)$。为所有$s$定义一个新的贪婪策略$\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s, a)$。
   c. **停止条件**：如果对于所有$s$，$\pi_{k+1}(s) = \pi_k(s)$，则停止并返回$\pi_{k+1}$和$V^{\pi_k}$。否则，继续下一次迭代（$k \leftarrow k+1$）。

因为有限MDP的策略数量是有限的，并且每个策略改进步骤都会产生一个严格更好或相等的策略，所以策略迭代保证在有限次迭代内收敛到最优策略$\pi^*$。

> 策略迭代过程在评估当前策略和根据评估结果贪婪地改进策略之间交替进行。

策略迭代的一个潜在缺点是策略评估步骤计算量较大，需要多次遍历状态空间，直到$V^\pi$精确收敛。

### 价值迭代（VI）

价值迭代提供了一种不同的方式。与在完整策略评估和改进之间交替不同，价值迭代通过重复应用贝尔曼*最优性*备份算子来直接迭代计算最优价值函数$V^*$。

回想一下$V^*$的贝尔曼最优性方程：


$$
V^*(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]
$$


价值迭代将这个方程转化为一个更新规则。它从初始猜测$V_0(s)$（通常全为零）开始，并迭代地对其进行修正：


$$
V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_k(s')]
$$


此更新在每次迭代$k$中同步（或异步）应用于所有状态$s$。请注意，行动的最大化（策略改进方面）是如何直接纳入价值更新（策略评估方面）的。

#### 价值迭代算法

1. 为所有$s \in S$初始化$V_0(s)$（例如，$V_0(s) = 0$）。
2. 初始化阈值$\theta > 0$（一个小的正数）。
3. **重复**：
   a. $\Delta \leftarrow 0$
   b. 对于每个状态$s$：
   i. $v \leftarrow V_k(s)$
   ii. $V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_k(s')]$
   iii. $\Delta \leftarrow \max(\Delta, |v - V_{k+1}(s)|)$
   c. $k \leftarrow k+1$
   d. **停止条件**：直到$\Delta < \theta$。
4. **输出**：一个确定性策略$\pi^*$，使得$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_{k}(s')]$。

价值迭代保证收敛到最优价值函数$V^*$，因为贝尔曼最优性算子是一个收缩映射，确保价值估计在每次迭代中都逐渐接近真实的$V^*$。一旦找到$V^*$（或其近似值），可以通过执行单步策略改进来提取最优策略$\pi^*$，如最终输出步骤所说明。



![价值迭代收敛示例](plots/3349-0.json)



> 价值迭代过程中特定状态的价值函数收敛示例。该价值在迭代过程中逐渐接近最优价值$V^*(s)$。

### 策略迭代与价值迭代的比较

- **策略迭代**：在完整策略评估（可能需要多次遍历）和单步策略改进之间交替。保证在有限次策略更新中找到$\pi^*$，但评估过程可能较慢。
- **价值迭代**：每次迭代执行一次贝尔曼最优性备份。每次迭代的计算比策略迭代中的完整策略评估更简单（无需等待$V^\pi$收敛），但价值迭代可能需要更多的总迭代次数才能收敛到$V^*$。

实践中，价值迭代通常收敛更快，尤其当可能的策略数量非常大时。也存在一些变体，例如改进的策略迭代，它在评估阶段只执行固定次数的贝尔曼期望备份，而不是迭代直到完全收敛。

价值迭代和策略迭代都是在已知MDP中寻找最优策略的基本动态规划方法。它们阐明了使用贝尔曼备份来评估和改进策略的主要原理，这些是许多我们稍后将学习的无模型强化学习 (reinforcement learning)算法所采用的核心思想。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习的经典教材，对MDPs的动态规划算法（包括策略迭代和价值迭代）提供了清晰而全面的介绍。
- [Markov Decision Processes: Discrete Stochastic Dynamic Programming](https://www.wiley.com/en-us/Markov+Decision+Processes%3A+Discrete+Stochastic+Dynamic+Programming%2C+2nd+Edition-p-9780471727828) — Martin L. Puterman (2005)
  Publisher: John Wiley & Sons
  马尔可夫决策过程和动态规划算法的严谨且全面的数学参考书。
- [Dynamic Programming and Optimal Control (Vol. I & II)](https://www.athenasc.com/dpbook.html) — Dimitri P. Bertsekas (2017)
  Publisher: Athena Scientific
  动态规划原理和最优控制的权威性参考书，广泛涵盖价值迭代和策略迭代。
