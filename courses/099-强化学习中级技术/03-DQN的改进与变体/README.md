# 第 3 章：DQN的改进与变体

来源：[原章节](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants)

[返回课程目录](../README.md)

尽管深度Q网络（DQN）算法为将神经网络应用于强化学习任务提供了坚实的基础，但其原始形式仍有改进空间。一个显著的问题是，标准DQN倾向于高估动作值，$Q(s, a)$，这有时会导致次优策略和训练过程中的不稳定。

本章将通过引入DQN框架的关键增强措施来解决这些局限性。你将学到：

*   Q学习和标准DQN中Q值高估问题的起因。
*   **双DQN (DDQN)** 如何修改更新规则，将动作选择与动作评估分离开来，从而减轻高估偏差。
*   **对偶网络架构**的原理，它将状态值函数$V(s)$和动作优势函数$A(s, a)$的估计区分开来。
*   这些改进如何结合使用，并对**优先经验回放**等技术进行简要介绍。

在本章结束时，你将明白这些变体如何在原始DQN的基础上构建，以创建更稳定、更有效的智能体，并且你将练习实现双DQN。

## 小节

- 1. [Q-学习中的估值过高问题](01-Q-%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E4%BC%B0%E5%80%BC%E8%BF%87%E9%AB%98%E9%97%AE%E9%A2%98.md)
- 2. [双重DQN (DDQN)](02-%E5%8F%8C%E9%87%8DDQN%20%28DDQN%29.md)
- 3. [对偶网络架构](03-%E5%AF%B9%E5%81%B6%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
- 4. [DQN改进的结合](04-DQN%E6%94%B9%E8%BF%9B%E7%9A%84%E7%BB%93%E5%90%88.md)
- 5. [优先经验回放 (简要概述)](05-%E4%BC%98%E5%85%88%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%20%28%E7%AE%80%E8%A6%81%E6%A6%82%E8%BF%B0%29.md)
- 6. [实践：实现双DQN](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%8F%8CDQN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/quiz)
