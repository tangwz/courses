# 第 2 章：马尔可夫决策过程 (MDPs)

来源：[原章节](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps)

[返回课程目录](../README.md)

在智能体、环境及其互动这些核心思想之上，我们现在需要一个正式的框架来描述强化学习所解决的问题。本章将介绍马尔可夫决策过程 (MDPs)，这是一种对序贯决策问题进行建模的常用数学工具，这类问题中的结果部分随机，部分受决策者控制。

你将了解如何定义一个MDP的主要组成部分：
*   环境可能处于的状态集合 ($S$)。
*   智能体可以采取的动作集合 ($A$)。
*   状态转移概率 ($P$)，它定义了环境的动态特性 ($P(s'|s, a)$)。
*   奖励函数 ($R$)，它规定了即时反馈 ($R(s, a, s')$)。
*   折扣因子 ($\gamma$)，它用来管理未来奖励的重要性。

我们将研究智能体的行为如何由策略 ($\pi$) 来定义，以及如何使用价值函数 ($V^\pi$ 和 $Q^\pi$) 来评估状态和状态-动作对的“优劣”。这将帮助我们理解MDP框架下强化学习的目标：即找到一个最优策略 ($\pi^*$)，使其最大化预期累积奖励。

## 小节

- 1. [序贯决策建模](01-%E5%BA%8F%E8%B4%AF%E5%86%B3%E7%AD%96%E5%BB%BA%E6%A8%A1.md)
- 2. [MDP的正式定义](02-MDP%E7%9A%84%E6%AD%A3%E5%BC%8F%E5%AE%9A%E4%B9%89.md)
- 3. [状态转移概率](03-%E7%8A%B6%E6%80%81%E8%BD%AC%E7%A7%BB%E6%A6%82%E7%8E%87.md)
- 4. [奖励函数](04-%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
- 5. [回报：未来累积奖励](05-%E5%9B%9E%E6%8A%A5%EF%BC%9A%E6%9C%AA%E6%9D%A5%E7%B4%AF%E7%A7%AF%E5%A5%96%E5%8A%B1.md)
- 6. [未来奖励的折现](06-%E6%9C%AA%E6%9D%A5%E5%A5%96%E5%8A%B1%E7%9A%84%E6%8A%98%E7%8E%B0.md)
- 7. [策略与价值函数 (Vπ, Qπ)](07-%E7%AD%96%E7%95%A5%E4%B8%8E%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0%20%28V%CF%80%2C%20Q%CF%80%29.md)
- 8. [寻找最优策略](08-%E5%AF%BB%E6%89%BE%E6%9C%80%E4%BC%98%E7%AD%96%E7%95%A5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps/quiz)
