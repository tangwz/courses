# 第 2 章：深度Q网络 (DQN)

来源：[原章节](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn)

[返回课程目录](../README.md)

第一章提到了将基本Q学习和SARSA应用于状态空间庞大或连续的问题所面临的难题。为每个可能的状态-动作对存储Q值在计算上变得不可行，并且使得泛化到未见过状态变得困难。

为了解决这个规模化问题，我们转向函数逼近。本章将介绍深度Q网络（DQN），这是一项重要进展，它使用深度神经网络来估计动作值函数$Q(s, a)$。这使得强化学习（RL）智能体能够在高维输入环境中有效学习，例如游戏屏幕图像。

我们将考察神经网络如何取代传统的Q表，以及使这种结合稳定有效的具体技术。你将了解到：

*   使用神经网络进行Q值逼近的基本思路。
*   DQN算法的核心架构和更新机制。
*   训练稳定性的重要方法：经验回放和目标网络。
*   为训练DQN定制的损失函数。
*   实现一个基本DQN智能体的分步指南。

完成本章后，你将明白DQN的运行原理，并通过为标准强化学习环境构建一个DQN智能体来获得实践经验。

## 小节

- 1. [函数近似的简介](01-%E5%87%BD%E6%95%B0%E8%BF%91%E4%BC%BC%E7%9A%84%E7%AE%80%E4%BB%8B.md)
- 2. [使用神经网络进行Q值近似](02-%E4%BD%BF%E7%94%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%BF%9B%E8%A1%8CQ%E5%80%BC%E8%BF%91%E4%BC%BC.md)
- 3. [DQN 算法架构](03-DQN%20%E7%AE%97%E6%B3%95%E6%9E%B6%E6%9E%84.md)
- 4. [经验回放机制](04-%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%E6%9C%BA%E5%88%B6.md)
- 5. [固定Q目标 (目标网络)](05-%E5%9B%BA%E5%AE%9AQ%E7%9B%AE%E6%A0%87%20%28%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%29.md)
- 6. [DQN训练的损失函数](06-DQN%E8%AE%AD%E7%BB%83%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 7. [动手实践：在CartPole上实现DQN](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8CartPole%E4%B8%8A%E5%AE%9E%E7%8E%B0DQN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/quiz)
