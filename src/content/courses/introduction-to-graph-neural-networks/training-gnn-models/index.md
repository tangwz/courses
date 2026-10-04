---
course: "introduction-to-graph-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models"
sourceId: 1319
chapter: "training-gnn-models"
title: "GNN 模型训练"
order: 4
description: "学习训练 GNN 的全过程，包括设置损失函数、图数据划分以及模型评估。"
hasQuiz: true
---

在了解 GNN 架构后，接下来的重点是学习如何训练这些模型。本章将介绍从确定 GNN 架构到完成特定图数据任务的全过程。

你将学习如何为节点分类等应用配置模型，包括将最终生成的节点嵌入传入分类层。接着，我们会说明如何选择合适的目标函数（如交叉熵损失）来衡量模型误差。本章的主要内容是构建标准的训练循环：包括执行前向传播、计算损失以及通过反向传播更新模型权重。

我们还会讲解图学习特有的操作。这包括数据划分时“转导式”（transductive）和“归纳式”（inductive）的区别，以及它们对模型评估的影响。你将学习使用准确率等标准指标来评估模型效果。最后，我们会介绍 dropout 等正则化方法，以提升模型的泛化能力。本章最后附带动手练习，让你应用上述步骤来训练之前构建的 GCN 模型。
