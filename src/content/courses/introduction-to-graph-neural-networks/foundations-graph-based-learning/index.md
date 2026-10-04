---
course: "introduction-to-graph-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning"
sourceId: 1316
chapter: "foundations-graph-based-learning"
title: "图学习基本原理"
order: 1
description: "学习图数据结构、常见的图机器学习任务，以及如何使用邻接矩阵和特征矩阵表示图以进行计算。"
hasQuiz: true
---

在构建图神经网络之前，我们需要先了解其处理的数据对象：图。本章讲解在机器学习环境下处理图结构数据的基本要点。

我们先从定义图的构成开始，并分析为什么 CNN 和 RNN 等传统神经网络不适合处理这类数据。接着，我们将介绍图上的主要机器学习任务，例如节点分类、链路预测和图分类。

本章的大部分篇幅用于说明如何为了计算而表示图。你将学习如何使用邻接矩阵 ($A$) 和节点特征矩阵 ($X$) 等标准格式来编码图的结构和属性。最后，我们将通过 NetworkX 库的简要介绍将这些想法付诸实践，使用该库加载并查看图数据集。
