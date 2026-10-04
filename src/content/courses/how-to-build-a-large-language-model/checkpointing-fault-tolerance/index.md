---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-19-checkpointing-fault-tolerance"
sourceId: 1097
chapter: "checkpointing-fault-tolerance"
title: "检查点和容错"
order: 19
description: "实施可靠的检查点保存机制，以应对故障并有效恢复长时间运行的大型语言模型训练作业。"
hasQuiz: false
---

训练大型语言模型是一个资源消耗大的过程，在大型计算集群上常常持续数天甚至数周。鉴于持续时间长，遇到硬件故障、软件问题或意外中断的可能性很大。如果没有保存进度的办法，这些中断可能导致训练从头开始，浪费大量计算资源和时间。

本章讨论了在大型训练中通过检查点保存实现容错的实际需要。我们将介绍定期保存训练作业完整状态所需的方法。您将学到：

*   训练状态由哪些部分组成（模型权重、优化器状态、调度器状态、数据加载器进度）。
*   如何在分布式训练环境中有效实现检查点保存，确保多个工作节点之间的一致性。
*   保存检查点的不同策略（同步与异步）及其相关的权衡取舍。
*   确定检查点保存频率以及管理可能很大的检查点文件存储需求时需要考虑的事项。
*   如何从已保存的检查点正确恢复训练作业，包括恢复模型、优化器和数据迭代状态的步骤。
