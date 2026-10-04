# 第 5 章：时序差分学习

来源：[原章节](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning)

[返回课程目录](../README.md)

在上一章中，我们审视了蒙特卡洛方法，这类方法只在一个完整的回合结束后才更新价值估计。这意味着必须等待最终结果确定，该回合的学习才能进行。

时序差分（TD）学习提供了一种不同的方式，它使得每一步之后都能进行更新。

TD方法直接从经验中学习，这一点与蒙特卡洛方法相似。然而，与蒙特卡洛不同的是，它们部分基于其他*当前已学到*的估计来更新价值估计，而无需等待回合的最终结果。这种技术被称为自举。它使得TD方法能够从不完整的回合中学习，并且在实际中通常比蒙特卡洛方法收敛更快。

本章将介绍时序差分学习的基础内容。我们将涉及：

*   用于估计状态价值函数 $V_\pi$ 的基本TD预测算法TD(0)。
*   与蒙特卡洛方法相比，TD学习的优势。
*   两种主要的TD控制算法：SARSA（状态-动作-奖励-状态-动作，一种同策略方法）和Q学习（一种常用的离策略方法）。我们将分析它们的更新规则和主要区别。
*   预期SARSA的介绍。
*   Q学习算法在解决简单控制任务中的实际实现。

在本章结束时，你将了解TD方法如何运作，并能够实现核心的TD算法，用于预测和控制问题。

## 小节

- 1. [从不完整的回合中学习](01-%E4%BB%8E%E4%B8%8D%E5%AE%8C%E6%95%B4%E7%9A%84%E5%9B%9E%E5%90%88%E4%B8%AD%E5%AD%A6%E4%B9%A0.md)
- 2. [TD(0) 预测：估计 Vπ](02-TD%280%29%20%E9%A2%84%E6%B5%8B%EF%BC%9A%E4%BC%B0%E8%AE%A1%20V%CF%80.md)
- 3. [TD学习相对于蒙特卡洛方法的优势](03-TD%E5%AD%A6%E4%B9%A0%E7%9B%B8%E5%AF%B9%E4%BA%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%96%B9%E6%B3%95%E7%9A%84%E4%BC%98%E5%8A%BF.md)
- 4. [SARSA：同策略TD控制](04-SARSA%EF%BC%9A%E5%90%8C%E7%AD%96%E7%95%A5TD%E6%8E%A7%E5%88%B6.md)
- 5. [Q学习：离策略TD控制](05-Q%E5%AD%A6%E4%B9%A0%EF%BC%9A%E7%A6%BB%E7%AD%96%E7%95%A5TD%E6%8E%A7%E5%88%B6.md)
- 6. [比较 SARSA 与 Q-学习](06-%E6%AF%94%E8%BE%83%20SARSA%20%E4%B8%8E%20Q-%E5%AD%A6%E4%B9%A0.md)
- 7. [期望SARSA](07-%E6%9C%9F%E6%9C%9BSARSA.md)
- 8. [动手实践：Q-学习的实现](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AQ-%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%AE%9E%E7%8E%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/quiz)
