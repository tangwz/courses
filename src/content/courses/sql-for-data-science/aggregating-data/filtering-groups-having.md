---
course: "sql-for-data-science"
chapter: "aggregating-data"
lesson: "filtering-groups-having"
sourceId: 1617
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/filtering-groups-having"
title: "使用 HAVING 过滤分组"
description: "了解如何使用 HAVING 子句过滤聚合函数的结果。"
order: 7
plots: []
sourceHash: "d859de5238f8ca8d7e5c0e9137b62677913a1b998a0bc5f800ddf6d00268a5bc"
sourceCorrections: []
---

SQL 的 `GROUP BY` 子句能够将数据聚合到汇总行中，例如可以找到每个客户的总订单数或每个产品类别的平均评分。但是，如果目标是仅查看符合特定条件的分组结果，例如只关注下了 *超过* 10 份订单的客户，或者平均评分 *高于* 4.0 的产品类别，该如何实现呢？

你可能最初会想到使用 `WHERE` 子句，就像我们在上一章中过滤单个行那样。然而，`WHERE` 有一个局限：它在聚合步骤发生 *之前* 过滤行。它对表中的原始数据进行操作，而不是对 `GROUP BY` 和 `COUNT()`、`AVG()` 或 `SUM()` 等聚合函数生成的汇总结果进行操作。

这就是 `HAVING` 子句的作用。

### 介绍 `HAVING` 子句

`HAVING` 子句专门用于在 `GROUP BY` 子句完成工作并计算出聚合函数之后 *过滤* 结果。它允许你对每个分组的汇总数据应用条件。

包含 `HAVING` 的查询基本结构如下：

```sql
SELECT
    column_name(s),
    aggregate_function(column_name)
FROM
    table_name
WHERE
    condition -- 可选：在聚合 *之前* 过滤行
GROUP BY
    column_name(s)
HAVING
    aggregate_condition -- 在聚合 *之后* 过滤分组
ORDER BY
    column_name(s); -- 可选：对最终结果进行排序
```

注意顺序：`WHERE` 在 `GROUP BY` 之前，`HAVING` 在 `GROUP BY` 之后。

### `WHERE` 与 `HAVING`：主要区别

理解何时使用 `WHERE` 以及何时使用 `HAVING` 对于编写涉及聚合的正确 SQL 查询来说很重要。

- **`WHERE` 子句：** 在行被分组和聚合 *之前* 过滤单个行。你在这里使用表中原始列上的条件。
- **`HAVING` 子句：** 在分组被 `GROUP BY` 创建并由聚合函数汇总 *之后* 过滤整个分组。你在这里使用聚合函数（例如 `COUNT(*)`、`AVG(price)`）的结果或分组列本身的条件。

这样来想：`WHERE` 决定哪些配料放入碗中进行混合，而 `HAVING` 决定哪些完成的碗（分组）值得保留。

让我们用一个例子来说明。想象一个 `Orders` 表：

| OrderID | CustomerID | OrderTotal |
| --- | --- | --- |
| 1 | 101 | 50.00 |
| 2 | 102 | 120.00 |
| 3 | 101 | 75.00 |
| 4 | 103 | 30.00 |
| 5 | 102 | 80.00 |
| 6 | 101 | 60.00 |
| 7 | 102 | 200.00 |

**场景 1：查找下了超过 2 份订单的客户。**

我们需要统计每个客户的订单，然后根据该计数进行过滤。

```sql
SELECT
    CustomerID,
    COUNT(OrderID) AS NumberOfOrders
FROM
    Orders
GROUP BY
    CustomerID
HAVING
    COUNT(OrderID) > 2; -- 根据聚合计数过滤分组
```

结果：

| CustomerID | 订单数量 |
| --- | --- |
| 101 | 3 |
| 102 | 3 |

在这里，`GROUP BY CustomerID` 首先按客户对行进行分组。然后 `COUNT(OrderID)` 计算每个客户的订单数量（101：3 份订单，102：3 份订单，103：1 份订单）。最后，`HAVING COUNT(OrderID) > 2` 过滤这些分组结果，只保留下了超过 2 份订单的客户。使用 `WHERE COUNT(OrderID) > 2` 会导致错误，因为 `COUNT(OrderID)` 在 `WHERE` 子句处理之后才计算。

**场景 2：查找总消费超过 \$150 的客户。**

```sql
SELECT
    CustomerID,
    SUM(OrderTotal) AS TotalSpending
FROM
    Orders
GROUP BY
    CustomerID
HAVING
    SUM(OrderTotal) > 150.00; -- 根据聚合总和进行过滤
```

结果：

| CustomerID | 总消费 |
| --- | --- |
| 101 | 185.00 |
| 102 | 400.00 |

同样，`HAVING` 在为每个 `CustomerID` 分组计算出 `SUM(OrderTotal)` *之后* 过滤结果。

**场景 3：查找下了超过 1 份订单的客户，但只考虑订单总额超过 \$50 的订单。**

在这里，我们需要同时使用 `WHERE` 和 `HAVING`。

```sql
SELECT
    CustomerID,
    COUNT(OrderID) AS NumberOfQualifyingOrders,
    SUM(OrderTotal) AS TotalQualifyingSpending
FROM
    Orders
WHERE
    OrderTotal > 50.00 -- 在分组 *之前* 过滤单个订单
GROUP BY
    CustomerID
HAVING
    COUNT(OrderID) > 1; -- 在聚合 *之后* 过滤分组
```

让我们追踪一下：

1. **`WHERE OrderTotal > 50.00`**：OrderID 为 1、2、3、5、6、7 的行被保留。行 4（OrderTotal = 30.00）被丢弃。
2. **`GROUP BY CustomerID`**：剩余的行被分组：
   - CustomerID 101: Orders 1, 3, 6
   - CustomerID 102: Orders 2, 5, 7
3. **聚合函数**：为每个分组计算 `COUNT(OrderID)` 和 `SUM(OrderTotal)`：
   - CustomerID 101: Count = 3, Sum = 50.00 + 75.00 + 60.00 = 185.00
   - CustomerID 102: Count = 3, Sum = 120.00 + 80.00 + 200.00 = 400.00
4. **`HAVING COUNT(OrderID) > 1`**：两个分组都满足此条件（3 > 1）。

结果：

| CustomerID | 符合条件的订单数量 | 符合条件的消费总额 |
| --- | --- | --- |
| 101 | 3 | 185.00 |
| 102 | 3 | 400.00 |

### 操作顺序的可视化

为了巩固区别，请考虑 SQL 处理这些子句的顺序：

> SQL 子句逻辑处理顺序的简化视图。`WHERE` 早期作用于单个行，而 `HAVING` 稍后作用于聚合分组。

如图所示，`WHERE` 在 `GROUP BY` 和聚合函数计算 *之前* 操作，而 `HAVING` 在之后操作。

总结来说，当你需要根据应用于 `GROUP BY` 创建的分组的聚合函数（`COUNT`、`SUM`、`AVG`、`MIN`、`MAX`）结果来过滤结果时，请使用 `HAVING`。如果你需要在聚合 *之前* 根据原始行中的值进行过滤，请使用 `WHERE`。有时，如第三个例子所示，你可能需要在同一个查询中同时使用 `WHERE` 和 `HAVING`，以在不同阶段实现所需的过滤。

## 参考资料

- [SELECT Statement](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2023)
  Publisher: PostgreSQL Global Development Group
  解释了PostgreSQL中SELECT语句（包括GROUP BY和HAVING子句）、它们的语法以及查询执行顺序。
- [Learning SQL: Master the Art of Database Interaction](https://www.oreilly.com/library/view/learning-sql/9781492076036/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  提供了SQL的详细解释，其中包含关于GROUP BY和HAVING子句的章节，通过示例阐明了它们的区别和用法。（第4版）
