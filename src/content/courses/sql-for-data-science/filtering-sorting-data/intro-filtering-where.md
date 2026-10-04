---
course: "sql-for-data-science"
chapter: "filtering-sorting-data"
lesson: "intro-filtering-where"
sourceId: 1575
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/intro-filtering-where"
title: "WHERE 子句筛选数据简介"
description: "了解 WHERE 子句的目的和基本语法，以便根据条件筛选行。"
order: 1
plots: []
sourceHash: "61dd3881cc8ccab0173dc16c05b8d25012ae76e2df59a6f29d81200a820beb1c"
sourceCorrections: []
---

使用 `SELECT` 语句从表中获取数据是 SQL 中的一个基本步骤。然而，仅仅选择所有数据通常会导致信息量过大。想象一个包含成千上万甚至数百万客户记录的表，您很少会一次性需要所有这些记录。相反，目标通常是专注于某个特定数据子集，例如来自特定城市的客户、在某个日期之后下的订单，或者某个价格范围内的产品。

这就是 `WHERE` 子句的作用所在。它是 SQL 中用于 **筛选行** 的主要工具。可以将其看作是设置条件，每行数据都必须满足这些条件才能包含在您的查询结果中。如果一行数据满足 `WHERE` 子句中指定的条件，则保留它；否则，则丢弃它。

### 目的与位置

`WHERE` 子句允许您指定搜索条件。它会在任何分组或排序发生*之前*，筛选由 `FROM` 子句返回的行。在基本的 `SELECT` 语句中，`WHERE` 子句紧随 `FROM` 子句之后：

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

在这里，`condition` 代表一个逻辑表达式，对于 `table_name` 中的每一行，其评估结果为 TRUE、FALSE 或 UNKNOWN。只有 `condition` 评估结果为 TRUE 的行才会包含在最终的结果集中。

### 条件如何作用

条件通常涉及使用比较运算符（我们接下来会讲到）将列的值与特定值或另一列的值进行比较。例如，您可能希望找到所有年龄大于 30 岁的用户，或者所有总金额超过 \$100 的订单。

我们来看一个简单的 `Products` 表：

| 产品ID | 产品名称 | 类别 | 价格 |
| --- | --- | --- | --- |
| 1 | 笔记本电脑 | 科技 | 1200 |
| 2 | 键盘 | 科技 | 75 |
| 3 | 咖啡杯 | 家居 | 15 |
| 4 | 办公椅 | 办公 | 250 |
| 5 | 无线鼠标 | 科技 | 25 |

如果您只想查看“科技”类别的产品，您可以使用类似这样的 `WHERE` 子句：

```sql
SELECT ProductID, ProductName, Price
FROM Products
WHERE Category = 'Tech';
```

当数据库处理此查询时，它会查看 `Products` 表中的每一行：

1. 第 1 行：`Category` 是 'Tech'。条件 `Category = 'Tech'` 为 TRUE。保留该行。
2. 第 2 行：`Category` 是 'Tech'。条件 `Category = 'Tech'` 为 TRUE。保留该行。
3. 第 3 行：`Category` 是 'Home'。条件 `Category = 'Tech'` 为 FALSE。丢弃该行。
4. 第 4 行：`Category` 是 'Office'。条件 `Category = 'Tech'` 为 FALSE。丢弃该行。
5. 第 5 行：`Category` 是 'Tech'。条件 `Category = 'Tech'` 为 TRUE。保留该行。

结果输出为：

| 产品ID | 产品名称 | 价格 |
| --- | --- | --- |
| 1 | 笔记本电脑 | 1200 |
| 2 | 键盘 | 75 |
| 5 | 无线鼠标 | 25 |

请注意，只有符合 `WHERE Category = 'Tech'` 条件的行才会被返回。另请注意，像 'Tech' 这样的文本值在 SQL 中通常用单引号 (`' '`) 括起来。数值通常不需要引号。

`WHERE` 子句是 SQL 中用于数据分析的非常有用的部分，它让您能够精确地获取所需数据。在接下来的部分中，我们将查看可用于构建更复杂的筛选条件的各种运算符。

## 参考资料

- [SELECT Statement: WHERE Clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-WHERE) — PostgreSQL Global Development Group (2024)
  Publisher: PostgreSQL Global Development Group
  涵盖 SELECT 语句的基本语法和行为，特别详细说明了 PostgreSQL 中用于过滤行的 WHERE 子句。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd-edition/9781492062634/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本全面的 SQL 入门指南，包含专门章节讲解 WHERE 子句及其在数据过滤中的各种应用。
- [Databases: Relational Databases and SQL](https://www.edx.org/course/databases-relational-databases-and-sql) — Jennifer Widom (2013)
  斯坦福大学的入门课程，涵盖数据库基础概念，包括对 SQL SELECT 和 WHERE 子句的详细解释和实践练习。
