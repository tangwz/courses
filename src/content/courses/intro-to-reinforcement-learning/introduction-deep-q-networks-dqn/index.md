---
course: "intro-to-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn"
sourceId: 489
chapter: "introduction-deep-q-networks-dqn"
title: "深度Q网络(DQN)简介"
order: 7
description: "了解深度Q网络(DQN)的基本内容，它将Q学习与深度神经网络进行结合。"
hasQuiz: true
---

在前一章函数逼近的思想之上，我们现在将重点放在Q学习与深度神经网络的结合。这种结合，即深度Q网络(DQN)，使智能体能够学到有效的策略，即使是在高维状态空间的环境中，例如游戏中的原始像素输入。

本章将阐述DQN的工作方式。我们首先会讨论使用深度神经网络来逼近动作值函数$Q(s, a; \theta)$的动因，这里$\theta$代表网络参数。然后我们会分析在使用强化学习数据训练这些网络时可能出现的内在不稳定性，例如连续样本之间的关联性，以及训练过程中目标值不断变化的问题。你会学到两种主要方法来缓解这些问题：经验回放，它随机存储并抽取过去的转移数据；以及使用独立的、周期性更新的目标网络来提供稳定的Q值目标。在本章结束时，你将明白标准DQN算法的构成，以及其主要组成部分的设计原理。
