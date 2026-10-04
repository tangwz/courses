---
course: "intro-to-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning"
sourceId: 487
chapter: "temporal-difference-learning"
title: "时序差分学习"
order: 5
description: "了解时序差分（TD）学习，包括SARSA和Q学习算法。"
hasQuiz: true
---

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
