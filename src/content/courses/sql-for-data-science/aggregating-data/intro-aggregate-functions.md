---
course: "sql-for-data-science"
chapter: "aggregating-data"
lesson: "intro-aggregate-functions"
sourceId: 1597
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/intro-aggregate-functions"
title: "聚合函数简介"
description: "了解聚合函数在汇总多行数据方面的作用。"
order: 1
plots: []
sourceHash: "589e426820de820eb8825f75acad346334480bba4fd8440106b17192385cd702"
sourceCorrections: []
---

使用 `SELECT` 从表中获取特定行和列，使用 `WHERE` 筛选结果，以及使用 `ORDER BY` 排序是 SQL 的基本操作。虽然查看单独的行很有用，但数据分析通常需要从更高层面了解数据。数据分析师可能不再问“这个具体产品的价格是多少？”，而是需要问“所有产品的平均价格是多少？”或“昨天下了多少订单？”。

这就是**聚合函数**发挥作用的地方。可以把它们看作是特殊的 SQL 函数，用于对多行数据执行计算，最终返回一个单一的汇总值。它们不是对每行单独操作，而是处理多行数据并生成一个结果。

假设你有一个简单的 `Orders` 表：

| OrderID | CustomerID | OrderTotal | OrderDate |
| --- | --- | --- | --- |
| 1 | 101 | 55.00 | 2023-10-26 |
| 2 | 102 | 120.50 | 2023-10-26 |
| 3 | 101 | 75.00 | 2023-10-27 |
| 4 | 103 | 30.25 | 2023-10-27 |
| 5 | 102 | 90.75 | 2023-10-28 |

使用聚合函数，你可以回答如下问题：

- **表中有多少订单？** (这需要计算行数)。
- **所有订单的总价值是多少？** (这需要对 `OrderTotal` 列求和)。
- **平均订单价值是多少？** (计算 `OrderTotal` 列的平均值)。
- **最小订单总额是多少？** (找到 `OrderTotal` 中的最小值)。
- **最大订单总额是多少？** (找到 `OrderTotal` 中的最大值)。

这些函数是汇总数据和获取有用的信息的重要工具。它们允许你将大量详细信息压缩为简洁、有用的统计数据。

在接下来的部分中，我们将了解 SQL 提供的一些最常用和有用的聚合函数：

- `COUNT`: 用于计数行或值。
- `SUM`: 用于计算列中值的总和。
- `AVG`: 用于计算列中值的平均值。
- `MIN`: 用于查找列中的最小值。
- `MAX`: 用于查找列中的最大值。

我们将首先把这些函数应用于整个表或结果集，然后在本章后面，你将学习如何使用 `GROUP BY` 子句将它们应用于数据中的特定分组。目前，请重点理解聚合函数提供了一种从多行数据计算单一汇总值的方法。

## 参考资料

- [PostgreSQL: Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — The PostgreSQL Global Development Group (2024)
  PostgreSQL的官方文档，提供标准聚合函数的详细解释和示例。它作为理解其语法和行为的权威技术参考。
- [Learning SQL, 3rd Edition](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057567/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本广受推崇且实用的SQL学习指南。本书为聚合函数提供了清晰的解释和实践示例，适合初学者和希望巩固SQL知识的人士。
- [Aggregate Functions (Transact-SQL)](https://learn.microsoft.com/en-us/sql/t-sql/functions/aggregate-functions-transact-sql) — Microsoft Learn (2023)
  Publisher: Microsoft
  微软针对SQL Server的Transact-SQL提供的官方文档，详细说明了聚合函数的语法和用法。此资源提供了来自另一个主要关系数据库系统的不同视角。
