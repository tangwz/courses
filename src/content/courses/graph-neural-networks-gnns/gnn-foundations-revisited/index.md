---
course: "graph-neural-networks-gnns"
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-1-gnn-foundations-revisited"
sourceId: 623
chapter: "gnn-foundations-revisited"
title: "重温图神经网络基础"
order: 1
description: "回顾图论、消息传递和与进阶图神经网络理解相关的谱方法。"
hasQuiz: false
---

为有效构建和分析进阶图神经网络模型，扎实掌握其基本原理是必需的。本章将巩固基础理论，从适用于更复杂结构和方法的视角来审视它们。

我们将首先考察用于机器学习任务的图数据表示方法，不止于基本的邻接矩阵($A$)，还涵盖图拉普拉斯算子($L = D - A$)和节点特征编码策略。接下来，我们将详细了解泛化消息传递框架，其通常概括为：
$$
\mathbf{h}_v^{(k)} = \text{UPDATE}^{(k)} \left( \mathbf{h}_v^{(k-1)}, \text{AGGREGATE}^{(k)} \left( \{ \mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v) \} \right) \right)
$$
我们将分析其理论性质和关于图神经网络表达能力的局限性，尤其结合魏斯费勒-莱曼 (WL) 图同构测试。

此外，我们将建立理解谱图神经网络所需的谱图理论背景知识。这包括图傅里叶变换和图上的谱卷积原理。图信号处理中与滤波器设计和分析相关的原理也将介绍。到本章结束时，您将巩固对这些核心思想的理解，为您后续讨论的进阶结构和难题做好准备。
