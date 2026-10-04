---
course: "intro-to-reinforcement-learning"
chapter: "foundations-reinforcement-learning"
lesson: "states-actions-rewards"
sourceId: 1555
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning/states-actions-rewards"
title: "状态、动作与奖励"
description: "了解强化学习的基本要素：表示情境的状态、可用的动作以及作为反馈的奖励。"
order: 3
plots: []
sourceHash: "f1678cedd337e635578a0328eb5b741545b7dd76eee48f857cdcb02af1867e83"
sourceCorrections: []
---

我们来了解一下状态、动作和奖励的具体内容。这三个要素构成了几乎所有强化学习 (reinforcement learning)问题的核心反馈循环。在我们进一步研究智能体实际如何*学习*之前，清楚地理解它们是必要的。基于智能体与环境交互的设想，我们现在明确它们之间有哪些信息流动。

### 状态 ($S$)

**状态**是环境在特定时刻的快照。它包含智能体需要（或被允许）感知来做决定的信息。可以把它看作是当前的情境或局面。

时间步 $t$ 的特定状态表示为 $s_t$，这是其正式定义。环境可以处于的所有可能状态的集合称为**状态空间**，表示为 $S$。状态的性质根据问题的不同而有很大差异：

- **简单网格：** 状态可以就是智能体的坐标，比如 $(x, y)$。状态空间是有限且离散的。
- **棋盘游戏（国际象棋、围棋）：** 状态是棋盘上所有棋子的位置。虽然是离散的，但可能的状态数量非常庞大。
- **机器人技术：** 状态可能包括关节角度、速度、传感器读数（如激光雷达或摄像头数据）和目标位置。这通常涉及连续值。
- **视频游戏（Atari）：** 状态可以是屏幕的原始像素值，可能结合了游戏内部信息，如得分或剩余生命数。这通常是高维状态空间。

与状态相关的一个重要方面是**可观测性**。

- **完全可观测：** 智能体的观测直接对应于环境的真实状态。智能体知道所有相关的信息来做出最优决策。许多经典的强化学习 (reinforcement learning)问题都假定完全可观测性。
- **部分可观测：** 智能体只接收到关于真实潜在状态的不完整信息的观测。例如，一个只有前置摄像头的机器人不知道它身后有什么。处理部分可观测性通常需要不同的技术，比如记住过去的观测。

本课程的大部分内容，尤其是在下一章介绍马尔可夫决策过程（MDP）等核心内容时，我们将假定环境是完全可观测的，或者我们使用的状态表示包含了所有相关信息（满足马尔可夫性质）。

### 动作 ($A$)

根据当前状态 $s_t$，智能体选择一个**动作**，表示为 $a_t$。动作仅仅是智能体做出的一个决定，是其可用的选择之一。

智能体可以采取的所有可能动作的集合是**动作空间**，表示为 $A$。有时，可用的动作取决于当前状态，在这种情况下，我们可能会写成 $A(s_t)$ 来表示状态 $s_t$ 中有效动作的集合。与状态类似，动作可以是：

- **离散型：** 有限的、不同的选择集合。
  - 网格：`move_north`、`move_south`、`move_east`、`move_west`。
  - 国际象棋：移动兵e2到e4，移动马g1到f3，等等（一个庞大但有限的集合）。
  - Atari游戏：`joystick_left`、`joystick_right`、`button_fire`。
- **连续型：** 由实数值表示的动作，通常在一定范围内。
  - 机器人技术：对电机施加特定扭矩（例如，1.5 牛米），设置转向角度（例如，-0.2 弧度）。
  - 恒温器控制：设定目标温度（例如，21.5 摄氏度）。

动作空间的性质显著影响哪种强化学习 (reinforcement learning)算法最适合。

### 奖励 ($R$)

在智能体在状态 $s_t$ 中采取动作 $a_t$ 后，环境转换到新状态 $s_{t+1}$ 并提供一个数值**奖励**，表示为 $r_{t+1}$。这个奖励信号很重要；它是告诉智能体表现如何的主要反馈机制。

- **目的：** 奖励定义了强化学习 (reinforcement learning)问题的目标。智能体的目标不一定是为了获得最高的即时奖励，而是为了最大化其长期获得的*总累积奖励*。
- **性质：** 奖励是标量值（单个数字）。它们可以是正的、负的或零。
  - 网格：到达目标状态 +10，掉入坑中 -5，每走一步 -0.1（以鼓励效率）。
  - 游戏：赢了 +1，输了 -1，所有中间步骤为 0（稀疏奖励）。或者，捕获对手棋子或取得有利位置时给予少量正奖励。
  - 机器人控制：到达目标为正奖励，碰到障碍物或消耗过多能量为负奖励。

**奖励假设**是强化学习中的一个基本观点：它指出所有目标和意图都可以被视为收到的标量信号（奖励）累积和的期望值的最大化。设计一个有效的奖励函数通常是在实践中应用强化学习最具有挑战性的部分之一。设计不当的奖励函数可能导致意料之外或次优的智能体行为。例如，仅仅因为收集灰尘而奖励一个清洁机器人，可能会导致它为了再次收集灰尘而倾倒灰尘！

### 循环

这三个组成部分构成了智能体-环境交互循环的依据：

1. 智能体观测当前**状态** $s_t$。
2. 智能体选择一个**动作** $a_t$。
3. 环境处理该动作，转换到下一个状态 $s_{t+1}$，并向智能体返回一个标量**奖励** $r_{t+1}$。
4. 这个循环重复进行（ $t$ 递增）。

> 智能体观测状态，选择一个动作，该动作影响环境。环境然后向智能体提供下一个状态和奖励，完成交互循环的一个步骤。

理解状态、动作和奖励是正式定义强化学习 (reinforcement learning)问题的第一步。在下一章中，我们将看到这些要素以及环境的动态特性如何在马尔可夫决策过程（MDP）的框架内得到体现。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapter 3: The Reinforcement Learning Problem
  一本基础教材，全面介绍了强化学习问题中作为智能体与环境交互核心的状态、动作和奖励概念。
- [CS234: Reinforcement Learning (Winter 2019) - Lecture 2: Markov Decision Processes](https://www.youtube.com/watch?v=1oD0mD_J4uE&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=2) — Emma Brunskill (2019)
  Journal: Stanford University Course Materials; Publisher: Stanford University
  大学课程讲义，清晰地定义了状态、动作、奖励及其在马尔可夫决策过程中的作用。
- [6.S191: Introduction to Deep Learning (2021) - Lecture 9: Reinforcement Learning](https://www.youtube.com/watch?v=VMz6S7xS3dM) — Lex Fridman (2021)
  Journal: MIT OpenCourseWare
  麻省理工学院深度学习入门课程中的一节课，简明扼要地介绍了强化学习的基本组成部分，包括状态、动作和奖励。
