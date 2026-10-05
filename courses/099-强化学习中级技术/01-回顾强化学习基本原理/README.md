# 第 1 章：回顾强化学习基本原理

来源：[原章节](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited)

[返回课程目录](../README.md)

在处理更高级的技术之前，有必要确保我们对强化学习（RL）的主要原理有扎实的掌握。本章旨在对强化学习学科中的主要知识进行一次集中温习。

我们将简要回顾：
*   包含智能体、环境、状态、动作和奖励的标准强化学习框架。
*   马尔可夫决策过程 (MDP)，作为序列决策问题的数学表达。
*   价值函数的主要作用，特别是状态价值函数 $V(s)$ 和动作价值函数 $Q(s, a)$。
*   贝尔曼方程，它是许多强化学习算法的构建依据：
    $$ V(s) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s] $$
*   经典的表格方法，如 Q-学习 和 SARSA，以及它们的更新机制。

最后，我们将审视这些表格方法的固有局限性，特别是在处理大型或连续状态空间时。对这些限制的理解将为后续章节中介绍的函数近似方法做好铺垫。此次复习有助于我们建立共同的认知，以便继续学习深度Q网络和策略梯度方法。

## 小节

- 1. [强化学习问题设置](01-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98%E8%AE%BE%E7%BD%AE.md)
- 2. [马尔可夫决策过程 (MDP) 回顾](02-%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B%20%28MDP%29%20%E5%9B%9E%E9%A1%BE.md)
- 3. [价值函数与贝尔曼方程](03-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0%E4%B8%8E%E8%B4%9D%E5%B0%94%E6%9B%BC%E6%96%B9%E7%A8%8B.md)
- 4. [表格型求解方法：Q学习和SARSA](04-%E8%A1%A8%E6%A0%BC%E5%9E%8B%E6%B1%82%E8%A7%A3%E6%96%B9%E6%B3%95%EF%BC%9AQ%E5%AD%A6%E4%B9%A0%E5%92%8CSARSA.md)
- 5. [表格方法的局限性](05-%E8%A1%A8%E6%A0%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/quiz)
