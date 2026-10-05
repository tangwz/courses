# 第 1 章：强化学习基本原理

来源：[原章节](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning)

[返回课程目录](../README.md)

本章为理解强化学习 (RL) 奠定基础。我们将从定义强化学习开始，并将其与其他机器学习方法进行比较。您将了解到下列核心组成部分：做出决策的*智能体*、智能体运行所在的*环境*、描述情境的*状态*、智能体可采取的*动作*，以及作为反馈收到的*奖励*。

我们将考察智能体的行为如何由*策略*决定，以及学习过程如何通过*交互循环*展开。*回合式*任务和*持续性*任务之间的区别也将得到解释。最后，我们将讲解如何使用 Gymnasium 和 NumPy 等库搭建 Python 环境的实际步骤。掌握这些原理是构建能够学习最优行为的智能体的第一步。

## 小节

- 1. [什么是强化学习？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%EF%BC%9F.md)
- 2. [智能体与环境](02-%E6%99%BA%E8%83%BD%E4%BD%93%E4%B8%8E%E7%8E%AF%E5%A2%83.md)
- 3. [状态、动作与奖励](03-%E7%8A%B6%E6%80%81%E3%80%81%E5%8A%A8%E4%BD%9C%E4%B8%8E%E5%A5%96%E5%8A%B1.md)
- 4. [策略：将状态映射到动作](04-%E7%AD%96%E7%95%A5%EF%BC%9A%E5%B0%86%E7%8A%B6%E6%80%81%E6%98%A0%E5%B0%84%E5%88%B0%E5%8A%A8%E4%BD%9C.md)
- 5. [强化学习工作流程：交互循环](05-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%EF%BC%9A%E4%BA%A4%E4%BA%92%E5%BE%AA%E7%8E%AF.md)
- 6. [强化学习任务类型：回合制与连续制](06-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%BB%BB%E5%8A%A1%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%9B%9E%E5%90%88%E5%88%B6%E4%B8%8E%E8%BF%9E%E7%BB%AD%E5%88%B6.md)
- 7. [强化学习与其他学习类型的比较](07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%85%B6%E4%BB%96%E5%AD%A6%E4%B9%A0%E7%B1%BB%E5%9E%8B%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 8. [为强化学习搭建Python环境](08-%E4%B8%BA%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%90%AD%E5%BB%BAPython%E7%8E%AF%E5%A2%83.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning/quiz)
