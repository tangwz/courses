---
course: "sql-for-data-science"
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data"
sourceId: 481
chapter: "aggregating-data"
title: "数据聚合"
order: 4
description: "学习如何使用 SQL 聚合函数如 COUNT、SUM、AVG、MIN、MAX 进行计算和汇总数据。"
hasQuiz: true
---

通常，单行数据所提供的信息不如从中得出的汇总信息有价值。在获取、筛选和排序数据之后，下一步通常是对分组数据计算汇总统计量。本章将介绍 SQL 聚合函数，它们是执行这些计算的工具。

您将学习如何:

*   使用 `COUNT`、`SUM`、`AVG`、`MIN` 和 `MAX` 等常见聚合函数，计算数据中的总和、平均值并找出边界值。
*   使用 `GROUP BY` 子句对具有共同特征的行进行分组，从而可以为每个不同的组计算聚合值。例如，计算每位客户的平均订单总值 $AVG(order\_total)$。
*   使用 `HAVING` 子句根据聚合值筛选这些分组后的结果，该子句在聚合操作完成后执行。
