---
course: "intermediate-reinforcement-learning"
chapter: "rl-fundamentals-revisited"
lesson: "rl-problem-setup-review"
sourceId: 5081
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-1-rl-fundamentals-revisited/rl-problem-setup-review"
title: "强化学习问题设置"
description: "回顾强化学习中的智能体、环境、状态、行动、奖励和策略。"
order: 1
plots: []
sourceHash: "87cced5939161d7b895fac2ad47352b186acfe830cd71e5ade71cc2547d710a5"
sourceCorrections: []
---

让我们先回顾一下强化学习 (reinforcement learning)问题的基本结构。正如章节引言中所述，强化学习本质上是通过交互来学习的。这种强化学习的设置主要包含两个构成部分：**智能体**和**环境**。

- **智能体**是学习者或决策者。可以将其看作您正在尝试构建的算法或控制器。它观察环境并决定采取哪些行动。
- **环境**是智能体与之交互的一切外部事物。它代表智能体需要解决的问题或任务。环境通过改变其状态和提供反馈来响应智能体的行动。

这种交互以一系列离散的时间步进行，$t = 0, 1, 2, ...$。在每个时间步 $t$，智能体观察环境的当前状况，这被表示为一个**状态**，记为 $S_t$。状态是对环境的描述，为智能体做出决策提供所需信息。这可以是棋盘上棋子的位置，也可以是机器人的传感器读数，或游戏屏幕的像素数据。

根据观察到的状态 $S_t$，智能体从一组可用行动中选择一个**行动** $A_t$。行动是智能体可以做出以影响环境的选择。示例包括移动机器人手臂、在游戏中做出一步，或调整恒温器。

在智能体于状态 $S_t$ 执行行动 $A_t$ 后，环境在下一个时间步转换到新状态 $S_{t+1}$。伴随新状态，环境向智能体提供一个数值**奖励** $R_{t+1}$。这个奖励信号表示在状态 $S_t$ 下采取的行动 $A_t$ 在短期内效果如何。正奖励通常对应有利结果，而负奖励（或成本）表示不利结果。

智能体的行为由其**策略**定义，记为 $\pi$。策略是智能体的应对方式，它将状态映射到行动。它决定了智能体在特定状态下选择哪个行动。策略可以是确定性的，写作 $a = \pi(s)$，这意味着它对于给定状态总是输出相同的行动。或者它可以是随机的，写作 $\pi(a|s) = P(A_t = a | S_t = s)$，指定在状态 $s$ 下采取每个可能行动 $a$ 的概率。

智能体的最终目标不仅仅是使即时奖励 $R_{t+1}$ 达到最高，而是使其长期累积的奖励总额达到最高。从时间步 $t$ 开始的这种累积奖励，通常称为**回报**，记为 $G_t$。一种常用表述是折扣回报：


$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$


这里，$\gamma$ 是**折扣因子**，一个介于 0 和 1 之间的数字 ($0 \le \gamma \le 1$)。它决定了未来奖励的现值。接近 0 的 $\gamma$ 使智能体“短视”，主要关注即时奖励，而接近 1 的 $\gamma$ 使智能体“有远见”，追求长期的高额总奖励。折扣因子还有助于使可能无限持续的任务中的回报保持有限。

这种状态-行动-奖励的连续循环构成了强化学习中的主要交互回路。

> 智能体与环境的交互循环。智能体观察状态，选择行动，环境转换到新状态并提供奖励，智能体随后观察奖励以指导未来的行动。

理解这些基本构成部分——智能体、环境、状态、行动、奖励和策略——是很有必要的。我们将在后续章节中讨论的算法，如Q学习、DQN、REINFORCE和Actor-Critic方法，都遵循此框架运作，为智能体学习一个能使其期望回报达到最高的有效策略提供了不同途径。对于大型问题，直接在表格中表示策略或价值函数的局限性，我们接下来会提及，这促使人们需要使用本课程的核心内容——函数逼近技术。

## 参考资料

- [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  介绍了RL的基础概念，包括智能体-环境接口、状态、动作、奖励、策略和回报，定义了问题设置。
- [CS234: Reinforcement Learning (Spring 2024 Lecture 1: Introduction, The RL Problem)](https://www.youtube.com/watch?v=R_0S_4E7a4E&list=PLo_uXvL_9g7XwGv2rJ3n8g5Q4P_8x_9t_) — Emma Brunskill (2024)
  Journal: Stanford University Course Materials; Publisher: Stanford Online (via YouTube)
  提供了RL问题设置的全面学术介绍，在一个结构化的课程背景下描述了智能体、环境、状态、动作、奖励、策略和回报的概念。
- [Spinning Up in Deep RL: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — Joshua Achiam (2018)
  Publisher: OpenAI
  一个定义了RL关键概念（如智能体-环境接口、状态、动作、奖励和策略）的资源，帮助实践者理解其设置。
- [Dynamic Programming and Markov Processes](https://mitpress.mit.edu/books/dynamic-programming-and-markov-processes) — Ronald A. Howard (1960)
  Publisher: MIT Press
  一篇开创性出版物，介绍了马尔可夫决策过程，这是强化学习问题设置的数学框架，讨论了状态转移、动作和最优策略。
