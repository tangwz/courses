---
course: "essential-numpy-pandas"
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas"
sourceId: 511
chapter: "grouping-aggregating-data-pandas"
title: "数据分组与聚合"
order: 9
description: "学习使用Pandas的`groupby()`方法进行“分拆-应用-组合”策略，以聚合数据并计算各组的汇总统计量。"
hasQuiz: true
---

在学习了Pandas的数据加载、清洗和选择之后，我们现在将转向一种数据分析的基本方法：按类别汇总数据。本章主要介绍分组操作。

您将学习使用“分拆-应用-组合”方法，这是数据聚合的一种常见模式。具体来说，您将学习：

*   使用`groupby()`方法根据列值对DataFrame进行分段。
*   对这些分段应用`sum()`、`mean()`、`count()`、`min()`和`max()`等聚合函数。
*   使用`.agg()`方法同时执行多个聚合操作。
*   根据多列对数据进行分组，以创建分层汇总。
*   在应用`groupby()`之后访问单个组及其相关数据。

学完本章后，您将能够有效地分段处理数据，并计算出DataFrame中不同组的有意义的汇总统计量。
