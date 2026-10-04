---
course: "advanced-reinforcement-learning"
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited"
sourceId: 712
chapter: "rl-foundations-revisited"
title: "基础重温与函数近似"
order: 1
description: "在处理高阶主题之前，回顾强化学习的核心理念，如马尔可夫决策过程、贝尔曼方程、时序差分学习、Q-学习 和 基本策略梯度。"
hasQuiz: false
---

为了给高阶强化学习方法奠定扎实基础，本章将对核心理念进行重点回顾。我们首先梳理马尔可夫决策过程 (MDP) 框架。将再次回顾诸如价值函数 ($V^\pi(s)$) 和行动价值函数 ($Q^\pi(s, a)$) 的贝尔曼方程等主要组成部分，以及动态规划求解方案（价值迭代和策略迭代）。接着，我们概述时序差分 (TD) 学习的基本算法，包括 Q-学习 和 SARSA，以及通过 REINFORCE 算法介绍策略梯度方法的基本原理。重要的是，本章引入了函数近似的理念，解释了其处理大型状态空间的必要性，并指出了结合离策略学习、函数近似和自举时可能出现的“致命三元组”带来的潜在不稳定性。这为您为本课程后续部分中会使用的深度学习方法做好了准备。
