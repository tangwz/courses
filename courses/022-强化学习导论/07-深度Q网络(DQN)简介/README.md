# 第 7 章：深度Q网络(DQN)简介

来源：[原章节](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn)

[返回课程目录](../README.md)

在前一章函数逼近的思想之上，我们现在将重点放在Q学习与深度神经网络的结合。这种结合，即深度Q网络(DQN)，使智能体能够学到有效的策略，即使是在高维状态空间的环境中，例如游戏中的原始像素输入。

本章将阐述DQN的工作方式。我们首先会讨论使用深度神经网络来逼近动作值函数$Q(s, a; \theta)$的动因，这里$\theta$代表网络参数。然后我们会分析在使用强化学习数据训练这些网络时可能出现的内在不稳定性，例如连续样本之间的关联性，以及训练过程中目标值不断变化的问题。你会学到两种主要方法来缓解这些问题：经验回放，它随机存储并抽取过去的转移数据；以及使用独立的、周期性更新的目标网络来提供稳定的Q值目标。在本章结束时，你将明白标准DQN算法的构成，以及其主要组成部分的设计原理。

## 小节

- 1. [Q学习与深度学习的结合](01-Q%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E7%BB%93%E5%90%88.md)
- 2. [强化学习中神经网络的难题](02-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 3. [经验回放机制](03-%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%E6%9C%BA%E5%88%B6.md)
- 4. [固定Q目标 (目标网络)](04-%E5%9B%BA%E5%AE%9AQ%E7%9B%AE%E6%A0%87%20%28%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%29.md)
- 5. [DQN 算法结构](05-DQN%20%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md)
- 6. [DQN 的网络结构设计考量](06-DQN%20%E7%9A%84%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E8%AE%BE%E8%AE%A1%E8%80%83%E9%87%8F.md)
- 7. [动手实践：构建一个基础DQN](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80DQN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/quiz)
