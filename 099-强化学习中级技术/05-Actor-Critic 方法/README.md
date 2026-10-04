# 第 5 章：Actor-Critic 方法

来源：[原章节](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods)

[返回课程目录](../README.md)

在之前的章节中，我们回顾了基于价值的方法，例如学习动作价值的深度Q网络（DQN），以及直接优化策略的策略梯度方法，例如REINFORCE。这两种方法各有优缺点。基于价值的方法可以提高样本效率，但在连续动作空间中表现不佳。策略梯度方法可以自然地处理连续动作，但其梯度估计往往方差较大。

本章将介绍Actor-Critic方法，这是一类结合了这两种方法特点的算法。您将了解这些方法如何使用两个组成部分：

*   **Actor（执行者）**：负责选择动作，类似于策略梯度方法。它学习一个参数化的策略$\pi_\theta(a|s)$。
*   **Critic（评论者）**：负责评估Actor所采取的动作，类似于基于价值的方法。它学习一个价值函数，通常是状态价值函数$V_\phi(s)$或动作价值函数$Q_\phi(s, a)$。

我们将分析评论者的评估如何为执行者提供方差更小的学习信号，旨在与纯策略梯度方法相比，实现更稳定、更高效的训练。我们将学习优势Actor-Critic (A2C) 及其异步变体 (A3C) 等具体实现，侧重于它们的架构、更新规则和实际考量。到本章结束时，您将理解Actor-Critic方法背后的原理，以及它们如何解决早期技术的一些局限性。

## 小节

- 1. [结合策略和价值评估](01-%E7%BB%93%E5%90%88%E7%AD%96%E7%95%A5%E5%92%8C%E4%BB%B7%E5%80%BC%E8%AF%84%E4%BC%B0.md)
- 2. [Actor-Critic 架构概述](02-Actor-Critic%20%E6%9E%B6%E6%9E%84%E6%A6%82%E8%BF%B0.md)
- 3. [优势演员-评论家 (A2C)](03-%E4%BC%98%E5%8A%BF%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%20%28A2C%29.md)
- 4. [异步优势参与者-评价者算法 (A3C)](04-%E5%BC%82%E6%AD%A5%E4%BC%98%E5%8A%BF%E5%8F%82%E4%B8%8E%E8%80%85-%E8%AF%84%E4%BB%B7%E8%80%85%E7%AE%97%E6%B3%95%20%28A3C%29.md)
- 5. [Actor-Critic 实现的考量](05-Actor-Critic%20%E5%AE%9E%E7%8E%B0%E7%9A%84%E8%80%83%E9%87%8F.md)
- 6. [对比：REINFORCE 与 A2C/A3C](06-%E5%AF%B9%E6%AF%94%EF%BC%9AREINFORCE%20%E4%B8%8E%20A2C-A3C.md)
- 7. [实践：开发 A2C 实现](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%80%E5%8F%91%20A2C%20%E5%AE%9E%E7%8E%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/quiz)
