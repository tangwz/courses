---
course: "federated-learning"
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-1-federated-learning-foundations-revisited"
sourceId: 731
chapter: "federated-learning-foundations-revisited"
title: "联邦学习核心原理重温"
order: 1
description: "回顾联邦学习核心原理、异构性等挑战、威胁模型和评估指标，以进行进阶学习。"
hasQuiz: false
---

为能有效在此之上构建联邦学习 (FL) 知识，本章将重温其核心原理。我们首先回顾标准的联邦学习流程及其基本原理。

您将考察联邦环境中遇到的固有难题，特别是统计异构性（非独立同分布数据）和系统异构性（客户端硬件、网络及可用性上的差异）。我们将定义联邦优化中常用的数学目标函数，它通常表示为最小化跨客户端聚合的全局损失：

$$
F(w) = \sum_{k=1}^N p_k F_k(w)
$$

$F_k(w)$ 代表客户端 $k$ 的局部目标，而 $p_k$ 代表其贡献权重。

后续部分将比较同步和异步训练模式，分析联邦学习威胁模型内的潜在安全和隐私威胁，并讨论用于评估这些分布式系统性能、效率和公平性的重要衡量指标和方法。此次回顾为后续章节中介绍的进阶技术奠定必要铺垫。
