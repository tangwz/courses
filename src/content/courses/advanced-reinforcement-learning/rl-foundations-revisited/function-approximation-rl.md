---
course: "advanced-reinforcement-learning"
chapter: "rl-foundations-revisited"
lesson: "function-approximation-rl"
sourceId: 3357
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited/function-approximation-rl"
title: "强化学习中的函数逼近"
description: "函数逼近（线性与非线性）在大型状态/动作空间中的重要性。"
order: 6
plots: []
sourceHash: "5459f3bea056cb031f2ef4db853ce40a77b282dfb987024dc911ce02baeaf40c"
sourceCorrections: []
---

迄今为止回顾的方法，例如Q学习、SARSA和基本策略梯度，通常依赖于价值函数（如$V(s)$或$Q(s, a)$）或策略（$\pi(a|s)$）的表格表示。这意味着我们为每个单一状态或状态-动作对维护一个显式的值或概率。虽然在小型、离散的环境（如井字棋或小型网格世界）中简单且在理论上清晰，但随着问题复杂度的增加，这种方法很快变得不实用。

### 表格表示的局限

考虑状态空间非常大或连续的环境。

- **游戏：** 象棋估计有$10^{47}$个状态，围棋大约有$10^{170}$个。为这些游戏存储表格是不可能的。即使是Atari Pong这样更简单的视频游戏，如果天真地用像素值表示，也会有天文数字般的可能屏幕配置（状态）。
- **机器人：** 机械臂的状态可能由其关节的连续角度和速度决定。移动机器人的状态包括其连续位置和方向。将这些空间精细离散化会导致状态数量的激增，而粗略离散化则会丢失重要信息。
- **实际控制：** 控制工业流程、管理金融投资组合或优化物流网络涉及高维状态空间，通常包含连续变量。

状态或状态-动作对数量的这种激增通常被称为**维度灾难**。表格方法主要在两个方面受到其影响：

1. **内存：** 存储表格需要大量的内存。
2. **学习时间：** 每个状态或状态-动作对可能需要访问多次才能获得准确的估计。在大型空间中，智能体可能永远不会遇到大多数状态，使学习变得异常缓慢或不完整。

此外，表格方法无法**泛化**。了解一个状态并不能提供关于类似但未访问状态的信息。如果你得知某个特定的棋盘位置很差，表格方法无法推断出另一个略有不同但策略上等效的位置也很差，除非明确访问它。

### 引入参数 (parameter)化函数逼近

为应对这些局限，我们引入**函数逼近**。我们不再将每个状态或状态-动作对的显式值存储在表格中，而是使用一个参数化函数（通常用参数$\theta$表示）来*逼近*真实的价值函数或策略。

- **价值函数逼近：**
  - 状态价值函数：$V(s) \approx \hat{V}(s; \theta)$
  - 动作价值函数：$Q(s, a) \approx \hat{Q}(s, a; \theta)$
- **策略逼近：**
  - 随机策略：$\pi(a|s) \approx \pi_\theta(a|s)$
  - 确定性策略：$a = \mu(s) \approx \mu_\theta(s)$

这里，$\hat{V}$、$\hat{Q}$、$\pi_\theta$和$\mu_\theta$代表函数（如线性函数、神经网络 (neural network)、决策树等），其行为由参数向量 (vector)$\theta$决定。学习算法的目标是找到使逼近尽可能准确的参数$\theta$。

> 对比表格方法与函数逼近在表示价值函数或策略方面的异同。

### 函数逼近器的类型

强化学习 (reinforcement learning)中可以使用各种函数逼近器：

1. **线性函数逼近：** 这是最简单的形式。我们首先定义一个特征向量 (vector)$\phi(s)$来表示状态$s$（或者$\phi(s, a)$用于状态-动作对）。逼近随后是这些特征的线性组合：

   
   $$
   \hat{V}(s; \theta) = \theta^T \phi(s) = \sum_{i=1}^d \theta_i \phi_i(s)
   $$
   
   
   $$
   \hat{Q}(s, a; \theta) = \theta^T \phi(s, a) = \sum_{i=1}^d \theta_i \phi_i(s, a)
   $$
   

   这里，$d$是特征的数量。逼近的质量很大程度上取决于人工设计的特征$\phi$的质量。虽然简单且计算效率高，但线性方法可能缺乏表示复杂非线性价值函数或策略的能力。
2. **非线性函数逼近（神经网络 (neural network)）：** 深度神经网络（DNN）已成为现代强化学习中函数逼近的主要选择，促成了深度强化学习（DRL）这一方向。神经网络可以直接从原始输入（如屏幕像素或关节角度）中学习复杂的非线性关系，而无需手动进行特征工程。

   - **价值网络：** $\hat{V}(s; \theta)$或$\hat{Q}(s, a; \theta)$由带有参数 (parameter)$\theta$（权重 (weight)和偏置 (bias)）的神经网络表示。
   - **策略网络：** $\pi_\theta(a|s)$或$\mu_\theta(s)$由神经网络表示。对于离散动作，网络通常输出每个动作的概率（例如，使用softmax输出层）。对于连续动作，它可能输出分布的均值以及可能的方差（例如，高斯分布）。

   常见的架构包括用于向量输入的**多层感知器（MLP）**、用于图像输入的**卷积神经网络（CNN）**以及用于序列输入或部分可观环境的**循环神经网络 (RNN)（RNN，如LSTM或GRU）**。DNN的优势在于它们能够自动学习分层特征并逼近高度复杂的函数。

### 算法如何适应

强化学习 (reinforcement learning)算法通过使用基于梯度的优化方法来适应与函数逼近器协同。我们不再更新表格条目，而是更新参数 (parameter)$\theta$。

- **基于价值的方法（例如，Q学习）：** 在表格Q学习中，更新规则将$Q(s_t, a_t)$调整到目标值$y_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$。使用函数逼近$\hat{Q}(s, a; \theta)$时，我们将其视为一个监督学习 (supervised learning)问题。目标$y_t$（通常使用一个单独的、缓慢更新的*目标网络*$\theta^-$来计算以确保学习过程的稳定，这将在第2章中看到）成为输入$(s_t, a_t)$的“标签”。我们旨在最小化预测Q值$\hat{Q}(s_t, a_t; \theta)$与目标$y_t$之间的误差。一个常用的损失函数 (loss function)是均方贝尔曼误差（MSBE）：

  
  $$
  L(\theta) = \mathbb{E} \left[ (y_t - \hat{Q}(s_t, a_t; \theta))^2 \right]
  $$
  

  其中$y_t = r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; \theta^-)$。
  参数$\theta$随后使用梯度下降 (gradient descent)法根据此损失进行更新：$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$。
- **基于策略的方法（例如，REINFORCE）：** 正如前面回顾的，策略梯度定理给出了预期回报$J(\theta)$的梯度表达式：

  
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
  $$
  

  这里，$\pi_\theta(a|s)$是我们的参数化策略（例如，一个神经网络 (neural network)）。我们通过在环境中运行当前策略$\pi_\theta$生成采样轨迹$(\tau)$来估计此梯度。参数随后使用梯度上升法进行更新：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$。
- **Actor-Critic方法：** 这些方法将在第3章中详细介绍，它们对策略（actor）和价值函数（critic）都使用函数逼近。Critic有助于为策略更新提供更低方差的估计，它结合了基于价值和基于策略方法的思路。

### 优势与挑战

使用函数逼近带来显著益处：

- **可扩展性：** 使得强化学习 (reinforcement learning)能够应用于表格方法无效的大型或连续状态/动作空间。
- **泛化能力：** 允许智能体将已访问状态的知识泛化到类似但未见的状态。如果$\hat{V}(s_1; \theta)$很高，并且$s_2$在特征或网络表示上“接近”$s_1$，那么$\hat{V}(s_2; \theta)$也很可能很高。
- **效率：** 通过运用逼近器捕获的状态空间底层结构，可以潜在地更快地学习。深度学习 (deep learning)方法能够自动进行特征提取。

然而，它也带来新的难题：

- **失去收敛性保证：** 与表格方法不同，对最优价值函数或策略的收敛性并非总能得到保证，特别是在非线性逼近器和离策略数据的情况下。
- **不稳定性：** 学习过程可能变得不稳定或发散。当结合以下三个要素时，这个问题尤为突出：函数逼近、离策略学习（从由不同策略生成的数据中学习）和自举（基于其他估计更新估计，如在TD学习中）。这种组合被称为“致命三要素”，我们将在后文中说明。
- **超参数 (parameter) (hyperparameter)敏感性：** 性能可能对逼近器架构的选择、学习率、优化算法以及其他超参数高度敏感。

“函数逼近对于将强化学习应用于复杂问题不可或缺。理解如何有效地将其与核心强化学习算法结合，并知晓相关的挑战，是掌握高级强化学习技术的核心所在。后续章节将大量地在此思路基础上进行，特别是使用深度神经网络 (neural network)作为主要的函数逼近器。”

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本全面的教科书，涵盖了强化学习的基础概念，包括表格法、函数逼近（线性和非线性）、策略梯度方法以及诸如“致命三元组”等挑战。
- [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg and Demis Hassabis (2015)
  Journal: Nature; Publisher: Nature Publishing Group; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了深度Q网络（DQN），这是一项将深度神经网络与强化学习成功结合的开创性工作，利用经验回放和目标网络等技术来稳定函数逼近的学习过程。
- [Reinforcement Learning Lecture 6: Value Function Approximation](https://www.davidsilver.uk/teaching/) — David Silver (2015)
  Publisher: UCL
  来自备受推崇的强化学习课程的讲义，提供了对值函数逼近方法及其与强化学习算法结合的易于理解的概述。
