# 强化学习中级技术

来源：[强化学习中级技术](https://apxml.com/zh/courses/intermediate-reinforcement-learning)

在您已有的强化学习 (reinforcement learning)知识之上。本课程涵盖重要的中级方法，包括深度Q网络 (DQN)、策略梯度法和Actor-Critic算法。学习运用函数逼近和高级策略来处理更复杂的序列决策问题。包含实践操作指南。

预计学时：18 小时

先修要求：具有强化学习初步知识。

## 课程目录

### 1. [回顾强化学习基本原理](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/README.md)

- 1. [强化学习问题设置](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/01-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98%E8%AE%BE%E7%BD%AE.md)
- 2. [马尔可夫决策过程 (MDP) 回顾](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/02-%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B%20%28MDP%29%20%E5%9B%9E%E9%A1%BE.md)
- 3. [价值函数与贝尔曼方程](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/03-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0%E4%B8%8E%E8%B4%9D%E5%B0%94%E6%9B%BC%E6%96%B9%E7%A8%8B.md)
- 4. [表格型求解方法：Q学习和SARSA](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/04-%E8%A1%A8%E6%A0%BC%E5%9E%8B%E6%B1%82%E8%A7%A3%E6%96%B9%E6%B3%95%EF%BC%9AQ%E5%AD%A6%E4%B9%A0%E5%92%8CSARSA.md)
- 5. [表格方法的局限性](01-%E5%9B%9E%E9%A1%BE%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/05-%E8%A1%A8%E6%A0%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- [章节测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/quiz)

### 2. [深度Q网络 (DQN)](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/README.md)

- 1. [函数近似的简介](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/01-%E5%87%BD%E6%95%B0%E8%BF%91%E4%BC%BC%E7%9A%84%E7%AE%80%E4%BB%8B.md)
- 2. [使用神经网络进行Q值近似](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/02-%E4%BD%BF%E7%94%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%BF%9B%E8%A1%8CQ%E5%80%BC%E8%BF%91%E4%BC%BC.md)
- 3. [DQN 算法架构](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/03-DQN%20%E7%AE%97%E6%B3%95%E6%9E%B6%E6%9E%84.md)
- 4. [经验回放机制](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/04-%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%E6%9C%BA%E5%88%B6.md)
- 5. [固定Q目标 (目标网络)](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/05-%E5%9B%BA%E5%AE%9AQ%E7%9B%AE%E6%A0%87%20%28%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%29.md)
- 6. [DQN训练的损失函数](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/06-DQN%E8%AE%AD%E7%BB%83%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 7. [动手实践：在CartPole上实现DQN](02-%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C%20%28DQN%29/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8CartPole%E4%B8%8A%E5%AE%9E%E7%8E%B0DQN.md)
- [章节测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/quiz)

### 3. [DQN的改进与变体](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/README.md)

- 1. [Q-学习中的估值过高问题](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/01-Q-%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E4%BC%B0%E5%80%BC%E8%BF%87%E9%AB%98%E9%97%AE%E9%A2%98.md)
- 2. [双重DQN (DDQN)](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/02-%E5%8F%8C%E9%87%8DDQN%20%28DDQN%29.md)
- 3. [对偶网络架构](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/03-%E5%AF%B9%E5%81%B6%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
- 4. [DQN改进的结合](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/04-DQN%E6%94%B9%E8%BF%9B%E7%9A%84%E7%BB%93%E5%90%88.md)
- 5. [优先经验回放 (简要概述)](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/05-%E4%BC%98%E5%85%88%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%20%28%E7%AE%80%E8%A6%81%E6%A6%82%E8%BF%B0%29.md)
- 6. [实践：实现双DQN](03-DQN%E7%9A%84%E6%94%B9%E8%BF%9B%E4%B8%8E%E5%8F%98%E4%BD%93/06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%8F%8CDQN.md)
- [章节测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/quiz)

### 4. [策略梯度方法](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/README.md)

- 1. [基于价值方法的局限性](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/01-%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [策略直接参数化](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/02-%E7%AD%96%E7%95%A5%E7%9B%B4%E6%8E%A5%E5%8F%82%E6%95%B0%E5%8C%96.md)
- 3. [策略梯度定理](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/03-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E5%AE%9A%E7%90%86.md)
- 4. [REINFORCE 算法](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/04-REINFORCE%20%E7%AE%97%E6%B3%95.md)
- 5. [理解策略梯度中的方差](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/05-%E7%90%86%E8%A7%A3%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E4%B8%AD%E7%9A%84%E6%96%B9%E5%B7%AE.md)
- 6. [方差减少的基线](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/06-%E6%96%B9%E5%B7%AE%E5%87%8F%E5%B0%91%E7%9A%84%E5%9F%BA%E7%BA%BF.md)
- 7. [动手实践：实现REINFORCE算法](04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0REINFORCE%E7%AE%97%E6%B3%95.md)
- [章节测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/quiz)

### 5. [Actor-Critic 方法](05-Actor-Critic%20%E6%96%B9%E6%B3%95/README.md)

- 1. [结合策略和价值评估](05-Actor-Critic%20%E6%96%B9%E6%B3%95/01-%E7%BB%93%E5%90%88%E7%AD%96%E7%95%A5%E5%92%8C%E4%BB%B7%E5%80%BC%E8%AF%84%E4%BC%B0.md)
- 2. [Actor-Critic 架构概述](05-Actor-Critic%20%E6%96%B9%E6%B3%95/02-Actor-Critic%20%E6%9E%B6%E6%9E%84%E6%A6%82%E8%BF%B0.md)
- 3. [优势演员-评论家 (A2C)](05-Actor-Critic%20%E6%96%B9%E6%B3%95/03-%E4%BC%98%E5%8A%BF%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%20%28A2C%29.md)
- 4. [异步优势参与者-评价者算法 (A3C)](05-Actor-Critic%20%E6%96%B9%E6%B3%95/04-%E5%BC%82%E6%AD%A5%E4%BC%98%E5%8A%BF%E5%8F%82%E4%B8%8E%E8%80%85-%E8%AF%84%E4%BB%B7%E8%80%85%E7%AE%97%E6%B3%95%20%28A3C%29.md)
- 5. [Actor-Critic 实现的考量](05-Actor-Critic%20%E6%96%B9%E6%B3%95/05-Actor-Critic%20%E5%AE%9E%E7%8E%B0%E7%9A%84%E8%80%83%E9%87%8F.md)
- 6. [对比：REINFORCE 与 A2C/A3C](05-Actor-Critic%20%E6%96%B9%E6%B3%95/06-%E5%AF%B9%E6%AF%94%EF%BC%9AREINFORCE%20%E4%B8%8E%20A2C-A3C.md)
- 7. [实践：开发 A2C 实现](05-Actor-Critic%20%E6%96%B9%E6%B3%95/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%80%E5%8F%91%20A2C%20%E5%AE%9E%E7%8E%B0.md)
- [章节测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/quiz)

## 学习目标

- **函数逼近**：理解为何及如何使用强化学习中的函数逼近器（如神经网络）。
- **深度Q网络 (DQN)**：实现并理解DQN的组成部分，包括经验回放和目标网络。
- **DQN变体**：学习DQN的改进，例如双重DQN和对决DQN。
- **策略梯度法**：掌握策略梯度背后的理论，并实现REINFORCE算法。
- **Actor-Critic方法**：理解Actor-Critic算法（例如A2C/A3C）的架构和优势。
- **算法实现**：获得实现这些中级强化学习算法的实践经验。
