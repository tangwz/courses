---
course: "graph-neural-networks-gnns"
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities"
sourceId: 629
chapter: "gnn-training-complexities"
title: "应对GNN训练中的复杂性"
order: 3
description: "应对GNN训练中的挑战，包括过平滑、过挤压、标签传播问题和可扩展性。"
hasQuiz: false
---

了解GNN架构很重要，但有效地训练它们，尤其是在大规模数据上，会带来独特的挑战。诸如过平滑（节点表示不理想地趋于一致）和过挤压（限制了信息在图上的传播）等问题，会大幅降低性能。在非常大的图上训练GNN还会带来大量计算和内存负担。

本章将侧重于讨论这些实际的训练复杂性。我们将考察过平滑和过挤压等问题的理论原理，并讨论应对它们的方法，例如架构修改和特定的训练技术。您还将学习将GNN训练扩展到海量数据集的策略，包括邻居采样（例如GraphSAGE）、图采样（例如GraphSAINT）和图聚类方法（例如Cluster-GCN）。最后，我们将提及这些模型的相关优化考量。
