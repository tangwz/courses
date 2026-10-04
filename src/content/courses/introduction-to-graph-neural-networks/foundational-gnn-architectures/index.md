---
course: "introduction-to-graph-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures"
sourceId: 1318
chapter: "foundational-gnn-architectures"
title: "GNN 基本架构"
order: 3
description: "学习三种基本图神经网络架构的技术细节：图卷积网络 (GCN)、GraphSAGE 和图注意力网络 (GAT)。"
hasQuiz: true
---

前一章确立了通用消息传递框架，为图神经网络的运行方式提供了蓝图。本章将从抽象的公式转向具体的知名架构，将这些原理付诸实践。我们将分析聚合函数和更新函数的不同选择如何产生具有不同行为和性能特征的模型。

你将学习三种基本模型背后的机制：

*   **图卷积网络 (GCN)：** 一种流行且高效的架构，通过简化的谱方法将卷积逻辑适配到图结构中。
*   **GraphSAGE：** 一种归纳式模型，通过学习通用的聚合函数并采用邻居采样，使其能够为训练期间未见过的节点生成嵌入。
*   **图注意力网络 (GAT)：** 一种引入掩码自注意力机制的架构，用于为邻域内的不同节点分配不同的权重。

针对每种模型，我们将讲解其数学公式，并讨论其主要优缺点。本章最后包含一个动手练习，通过从零开始构建 GCN，将理论公式直接转化为可运行的代码。
