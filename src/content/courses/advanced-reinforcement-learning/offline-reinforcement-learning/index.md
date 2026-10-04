---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning"
sourceId: 728
chapter: "offline-reinforcement-learning"
title: "离线强化学习"
order: 7
description: "了解离线强化学习（批量强化学习），它完全通过固定数据集学习策略，无需与环境交互。解决分布偏移问题。"
hasQuiz: false
---

之前的章节主要关注通过与环境主动交互来提升能力的强化学习智能体。本章将转向**离线强化学习**，它常被称为批量强化学习（Batch RL）。在这里，面临的挑战是仅凭静态的、预先收集好的转移数据集来学习有效策略，而没有机会再与环境进行交互。

我们将首先阐述离线学习的动机，并强调它与之前讨论的在线和异策略设置之间的主要区别。离线强化学习的一个核心难题是**分布偏移**：即固定数据集（由某种行为策略 $ \pi_b $ 生成）中状态和动作的分布，与正在评估或学习的新策略所产生的分布不匹配。

本章将论述：
*   分布偏移的含义及其影响。
*   **离线异策略评估（OPE）**的方法，包括重要性采样及其在实践中遇到的难题。
*   专门设计用于缓解分布偏移的算法，例如**策略约束方法**（如BCQ），这些方法将学习到的策略限制在数据支持的动作上。
*   **值正则化技术**（如CQL），这些技术会惩罚数据集中未出现的动作所产生的不切实际的Q值。
*   与离线设置相关的实际实现细节。
