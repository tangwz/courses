---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies"
sourceId: 720
chapter: "advanced-exploration-strategies"
title: "进阶搜寻策略"
order: 4
description: "学习超越简单方法的精密搜寻技术，包括 UCB、汤普森采样、基于计数的方法和内在动机。"
hasQuiz: false
---

有效的强化学习需要在搜寻环境以发现更好的策略与利用现有知识以获得即时奖励之间取得谨慎的平衡。基本的搜寻方法，例如 $\epsilon$-贪心法，在涉及大型状态空间或稀疏反馈的复杂情境中往往力有不逮。本章介绍进阶搜寻技术，能够实现更高效、更有方向性的发现。

你将考察根植于管理不确定性的策略，包括上置信区间（UCB）方法和汤普森采样。我们还将研究基于计数的方法，这些方法鼓励访问较不熟悉的状态，以及内在动机技术，其中智能体根据预测误差（如ICM）、状态新颖性（如RND）或信息增益产生内部奖励。最后，将讨论参数空间噪声在搜寻中的应用。理解这些方法将为你提供工具，来设计能够应对困难搜寻难题的智能体。
