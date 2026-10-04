# 第 3 章：高级策略梯度与 Actor-Critic 方法

来源：[原章节](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic)

[返回课程目录](../README.md)

基础策略梯度方法，例如 REINFORCE，其梯度估计常存在高方差问题，导致学习缓慢或不稳定。本章介绍 Actor-Critic 方法，这是一类旨在应对此局限性的算法。其核心思想是维持两个组成部分：一个 *actor*（行动者），它学习策略 $\pi(a|s)$；以及一个 *critic*（评估者），它学习价值函数（例如 $V(s)$ 或 $Q(s,a)$）来评估 actor 的行动并提供低方差的梯度信号。

你将学习在此框架基础上的一些重要进展：

*   **方差降低**：例如使用基线以及 Advantage Actor-Critic (A2C/A3C) 来稳定策略更新的方法。
*   **优势估计**：理解广义优势估计 (GAE) 以平衡偏差与方差。
*   **连续控制**：例如深度确定性策略梯度 (DDPG) 等适用于连续动作空间的算法。
*   **策略优化稳定性**：确保更可靠策略改进的方法，具体而言是信任区域策略优化 (TRPO) 和近端策略优化 (PPO)。
*   **最大熵强化学习**：Soft Actor-Critic (SAC)，一种采用熵最大化以获得更好试探能力和稳定性的离策略方法。

在本章结束时，你将理解这些高级算法的原理，并准备好实现它们以解决更复杂的强化学习问题。

## 小节

- 1. [基本策略梯度面临的挑战](01-%E5%9F%BA%E6%9C%AC%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E9%9D%A2%E4%B8%B4%E7%9A%84%E6%8C%91%E6%88%98.md)
- 2. [行动者-评论者架构基本原理](02-%E8%A1%8C%E5%8A%A8%E8%80%85-%E8%AF%84%E8%AE%BA%E8%80%85%E6%9E%B6%E6%9E%84%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 3. [降低方差的基线](03-%E9%99%8D%E4%BD%8E%E6%96%B9%E5%B7%AE%E7%9A%84%E5%9F%BA%E7%BA%BF.md)
- 4. [优势 Actor-Critic (A2C) 和 A3C](04-%E4%BC%98%E5%8A%BF%20Actor-Critic%20%28A2C%29%20%E5%92%8C%20A3C.md)
- 5. [广义优势估计 (GAE)](05-%E5%B9%BF%E4%B9%89%E4%BC%98%E5%8A%BF%E4%BC%B0%E8%AE%A1%20%28GAE%29.md)
- 6. [深度确定性策略梯度 (DDPG)](06-%E6%B7%B1%E5%BA%A6%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%20%28DDPG%29.md)
- 7. [信任区域策略优化 (TRPO)](07-%E4%BF%A1%E4%BB%BB%E5%8C%BA%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%20%28TRPO%29.md)
- 8. [近端策略优化 (PPO)](08-%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%20%28PPO%29.md)
- 9. [软演员-评论家 (SAC)](09-%E8%BD%AF%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%20%28SAC%29.md)
- 10. [演员-评论家方法实现实践](10-%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%E6%96%B9%E6%B3%95%E5%AE%9E%E7%8E%B0%E5%AE%9E%E8%B7%B5.md)
