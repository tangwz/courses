---
course: "data-modeling-schema-design-analytics"
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-2-dimensional-modeling-core"
sourceId: 1390
chapter: "dimensional-modeling-core"
title: "维度建模核心"
order: 2
description: "掌握维度建模的基础知识，包括事实、维度和星型模型设计。"
hasQuiz: false
---

原始数据存储机制与有效分析所需的逻辑结构大不相同。尽管前一章涉及了存储层，但本节将侧重于数据的逻辑组织。维度建模提供了一种专门的数据构建方法，以支持高性能查询和直观的报告。

首要目的是将度量数据与描述性背景分离。您将学会区分存储量化指标的事实表，以及提供“谁、什么、何地、何时”等信息的维度表。此过程的一个核心部分是定义数据的“粒度”。明确定义的粒度确保收入或数量等指标在聚合时行为可预测。例如，在严格可加的事实表中，任何维度上的总和都可简单计算为 $Total = \sum x_i$。

我们将考察星型模型的特定属性，其中维度直接连接到中心事实表；并将其与雪花型模型进行对比，雪花型模型通过规范化维度表来减少冗余。通过这些课程，您将获得将业务需求转化为一种兼顾查询速度和维护要求的模式设计的能力。
