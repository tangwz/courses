# SQL JOINs 简介

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/intro-sql-joins)

[返回章节目录](README.md) · [返回课程目录](../README.md)

关系数据库中的数据通常分布在多个表中，以保持组织性和减少冗余。例如，您可能有一个表列出客户详情，而另一个表列出这些客户下的订单。为了分析客户的订单模式，您需要一种方法来关联这两个表中的信息。SQL 为此提供了一个强大的机制：`JOIN` 操作。

SQL 中的 `JOIN` 子句用于根据表之间相关列的值来组合来自两个或更多表的行。可以将其视为水平地拼接表，其中特定列中的值相匹配。这种关系通常通过使用主键和外键来建立，它们充当表之间的链接，确保数据的完整性。

> 通过 `CustomerID` 列连接 `Customers` (客户) 表和 `Orders` (订单) 表的视图。

尽管这个想法看起来简单，但连接表有不同的方式，取决于您希望在结果集中包含哪些行。SQL 提供多种 `JOIN` 操作类型以应对不同情况：

### 常用 SQL JOIN 类型

1. **`INNER JOIN`**：这是最常用的连接类型。它只返回在 *两个* 表中都满足连接条件的行。如果一个客户没有下订单，结果中将不会出现该客户或任何订单信息。同样，如果一个订单引用的 `CustomerID` 不存在于 `Customers` 表中（在适当的外键约束下不应发生这种情况），该订单将不会被包含。它根据连接列找出两个表的交集。
2. **`LEFT JOIN`** (或 `LEFT OUTER JOIN`)：此连接返回左表（`FROM` 子句中列出的第一个表）中的 *所有* 行以及右表（`JOIN` 子句中指定的表）中的匹配行。如果左表中的某行在右表中没有匹配项，结果仍将包含左表中的该行，但从右表选择的所有列将显示 `NULL` 值。当您希望获取主表（例如，所有客户）的所有信息，无论它们在次表（例如，订单）中是否有匹配记录时，这非常有用。
3. **`RIGHT JOIN`** (或 `RIGHT OUTER JOIN`)：这是 `LEFT JOIN` 的反向操作。它返回右表中的 *所有* 行以及左表中的匹配行。如果右表中的某行在左表中没有匹配项，结果将显示左表列的 `NULL` 值。
4. **`FULL OUTER JOIN`**：当在左表或右表中存在匹配时，此连接返回所有行。它实际上结合了 `LEFT JOIN` 和 `RIGHT JOIN` 的结果。任何表中特有的行都将被包含，并在没有找到匹配项的表的列中填充 `NULL` 值。

了解不同类型的连接是有益的，而 `INNER JOIN` 非常重要，在数据分析中广泛用于组合相关数据。本章将主要聚焦于 `INNER JOIN`。在接下来的小节中，我们将仔细查看编写 `INNER JOIN` 子句的语法、如何指定连接条件以及如何有效应用它来组合来自多个表的数据。

## 参考资料

- [PostgreSQL 16.2 Documentation: 7.2. Table Expressions - Joins Between Tables](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOINS) — PostgreSQL Global Development Group (2024)
  提供了领先关系数据库系统中SQL JOIN语法和行为的权威信息。
- [Practical SQL, 2nd Edition: A Complete Guide to Storytelling with Data](https://nostarch.com/practical-sql-2nd-edition/) — Anthony DeBarros (2022)
  Publisher: No Starch Press
  一本实用的指南，包含许多在数据分析中使用SQL的示例，并全面涵盖了各种JOIN类型。

---

[上一节](02-%E7%90%86%E8%A7%A3%E4%B8%BB%E9%94%AE%E5%92%8C%E5%A4%96%E9%94%AE.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20INNER%20JOIN.md)
