---
course: "introduction-to-graph-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric"
sourceId: 1320
chapter: "gnn-implementation-pytorch-geometric"
title: "使用 PyTorch Geometric 实现 GNN"
order: 5
description: "学习使用流行的 PyTorch Geometric (PyG) 库，高效地构建、训练和扩展图神经网络。"
hasQuiz: true
---

前面的章节说明了 GNN 的运行原理，包括从头开始实现消息传递层。虽然这很有启发性，但手动方法在构建和测试不同架构时并不实用。本章将转而使用 PyTorch Geometric (PyG)，这是一个简化图模型开发的专用库。PyG 提供了常用 GNN 层和数据处理程序的优化版本，使你能够把精力放在模型设计上，而非底层的实现细节。

在本章中，你将学会使用 PyTorch Geometric 的主要部分。我们将从 `Data` 对象开始，这是 PyG 表示整个图的方式。接着，你将看到如何加载标准图数据集，并使用 PyG 预置的层（例如 `GCNConv` 和 `GATConv`）来组建 GNN 模型。我们还会说明该库如何通过图的批量化处理来提升训练效率。

学完本章后，你将有能力为节点分类任务编写完整的训练和评估脚本。最后的动手操作环节将运用这些技能，在 Cora 引用网络这一标准基准数据集上构建 GNN。
