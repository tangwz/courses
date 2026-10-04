---
course: "data-modeling-schema-design-analytics"
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-3-advanced-dimension-handling"
sourceId: 1391
chapter: "advanced-dimension-handling"
title: "高级维度处理"
order: 3
description: "学习使用 SCDs 处理变化的数据，管理日期维度，并使用代理键。"
hasQuiz: false
---

上一章中，我们定义了星型和雪花型架构的结构组成部分。尽管这些模型对静态数据表现良好，但在生产环境中，属性会随时间变化。客户地址会变动，产品类别会重新组织，销售区域也会调整。假设数据不变的严格架构，最终会导致历史报告不准确。

本章侧重于管理变化的维度数据和特殊架构需求所需的机制。我们首先将源自操作系统的自然键与代理键分开。实施一个表示为整数 $k$ 的系统生成代理键，可以使您的分析模型摆脱源系统依赖，并提升连接性能。

接下来我们考察缓慢变化维度 (SCD)。您将学会区分需要 Type 1 更新（即数据被直接覆盖）和 Type 2 更新（即通过插入新行来保留历史）的场景。这种区分可以确保关于时间 $t$ 时的收入查询，准确反映维度属性在当时的状态，而非它们当前的状态。

本节以特殊维度模式作为结尾。我们将构建一个日期维度以支持非标准日历逻辑，例如会计季度，并实施角色扮演维度。这使得单个物理表可以在查询中服务于多种逻辑目的，例如使用一个日期表同时按 `order_date` 和 `ship_date` 过滤事实。
