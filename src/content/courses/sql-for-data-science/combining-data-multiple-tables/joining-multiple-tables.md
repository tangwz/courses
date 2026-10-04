---
course: "sql-for-data-science"
chapter: "combining-data-multiple-tables"
lesson: "joining-multiple-tables"
sourceId: 1634
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/joining-multiple-tables"
title: "连接多个表"
description: "扩展 `INNER JOIN` 的想法，以组合来自三个或更多相关表的数据。"
order: 5
plots: []
sourceHash: "2ccd4d3cc75cc1a7cbfadd833ba94fe50819c1d23a37db4990c449fbf9ef36ee"
sourceCorrections: []
---

`INNER JOIN` 是一种基本的 SQL 操作，它根据相关列（通常使用主键和外键）来合并两个表的行。然而，数据分析经常需要同时从*多于*两个表中获取信息。设想一下，你需要客户详情、他们的订单信息以及他们所订购产品的具体信息。这些信息可能存储在三个独立的表中：`Customers`、`Orders` 和 `Products`。

幸运的是，将 `INNER JOIN` 这个想法扩展到多个表是非常合理的。你基本上是把连接操作串联起来。你将第一个表连接到第二个表，然后将该操作的结果连接到第三个表，依此类推。每个 `JOIN` 操作都需要自己的 `ON` 条件来指明表之间如何关联。

### 多表连接的语法

这种模式涉及添加后续的 `INNER JOIN` 和 `ON` 子句。要连接三个表，比如 `table1`、`table2` 和 `table3`，结构如下所示：

```sql
SELECT
    -- 从任何已连接的表中选择列
    t1.columnA,
    t2.columnB,
    t3.columnC
FROM
    table1 AS t1  -- 从第一个表开始（使用别名！）
INNER JOIN
    table2 AS t2 ON t1.key_column = t2.foreign_key_t1 -- table1 和 table2 之间的连接条件
INNER JOIN
    table3 AS t3 ON t2.key_column = t3.foreign_key_t2; -- table2 和 table3 之间的连接条件
    -- 或者可能是：ON t1.key_column = t3.foreign_key_t1，取决于它们的关系
```

注意以下几点：

1. **表别名：** 如前一节所述，在连接多个表时，别名（`t1`、`t2`、`t3`）变得更有用，几乎是必需的。它们使查询保持可读性，并在表共享列名（如 `id` 列）时避免歧义。
2. **连接条件：** 每个 `INNER JOIN` 语句后都必须跟一个 `ON` 子句，指明如何将*新*表（第二次连接中的 `table3`）连接到已包含在连接操作中的表（`table1` 或 `table2`）。你需要识别每个连接中正确的主键/外键关系。

### 示例：客户、订单和产品

让我们考虑一个包含三个表的实际情况：

- `Customers`：包含客户信息（`CustomerID`，`CustomerName`）。
- `Orders`：包含订单详情（`OrderID`，`CustomerID`，`ProductID`，`OrderDate`）。`CustomerID` 关联到 `Customers`，`ProductID` 关联到 `Products`。
- `Products`：包含产品详情（`ProductID`，`ProductName`，`Price`）。

以下是这些表可能如何关联的可视化表示：

> 该图表显示 `Customers` 通过 `CustomerID` 连接到 `Orders`，`Products` 通过 `ProductID` 连接到 `Orders`。

现在，假设我们想要一个列表来显示客户的姓名、他们所订购产品的名称以及订单日期。我们需要来自所有三个表的数据。我们可以通过将 `Customers` 连接到 `Orders`，然后将该结果连接到 `Products` 来实现这一点：

```sql
SELECT
    c.CustomerName,
    p.ProductName,
    o.OrderDate
FROM
    Customers AS c
INNER JOIN
    Orders AS o ON c.CustomerID = o.CustomerID -- 连接 Customers 和 Orders
INNER JOIN
    Products AS p ON o.ProductID = p.ProductID; -- 将结果与 Products 连接
```

**说明：**

1. `FROM Customers AS c`：我们从 `Customers` 表开始，赋予它别名 `c`。
2. `INNER JOIN Orders AS o ON c.CustomerID = o.CustomerID`：我们将 `Customers`（`c`）与 `Orders`（`o`）连接，其中两个表中的 `CustomerID` 匹配。这将每个客户关联到其各自的订单。
3. `INNER JOIN Products AS p ON o.ProductID = p.ProductID`：然后我们获取第一次连接的结果（现在包含匹配的客户和订单数据），并将其与 `Products` 表（`p`）连接。当 `Orders` 表（`o`）中的 `ProductID` 与 `Products` 表（`p`）中的 `ProductID` 匹配时，连接便建立。这会将产品信息添加到对应的订单中。
4. `SELECT c.CustomerName, p.ProductName, o.OrderDate`：最后，我们从组合结果中选择所需的列，使用别名来指明每列来自哪个表（`c.CustomerName`、`p.ProductName`、`o.OrderDate`）。

这个查询有效地将来自三个不同表的相关信息整合到一个单一、连贯的结果集中。

### 注意事项

- **识别关联：** 在编写查询之前，务必梳理出表之间如何连接。一个表中的哪个外键连接到另一个表中的哪个主键？像上面的图表一样将其可视化，会非常有帮助。
- **连接顺序（对于 INNER JOIN）：** 对于 `INNER JOIN`，你列出表的顺序和执行连接的顺序通常不会改变最终输出（尽管在某些数据库系统中可能会影响性能）。然而，逻辑地构建连接（例如，按照 `Customers` -> `Orders` -> `Products` 这样的关系）通常会使查询更容易理解。
- **正确的连接条件：** 确保每个 `ON` 子句都使用正确的列来连接表。在错误的列上进行连接是一个常见错误，会导致错误的结果或意外的行组合。

连接多个表是一种强大的技巧，它允许你整合分散在数据库架构中的数据。通过链式连接 `INNER JOIN` 子句并仔细指明 `ON` 条件中的关系，你可以构建提供全面视图的查询，以完成你的数据分析任务。

## 参考资料

- [MySQL 8.0 Reference Manual :: 15.2.13 JOIN Clause](https://docs.oracle.com/en/mysql/mysql-server/8.0/refman/8.0/en/join.html) — MySQL Documentation Team (2023)
  Publisher: Oracle
  官方文档，详细介绍了SQL JOIN的各种类型、语法和用法，包括多表连接。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本实用的指南，通过示例清晰地解释了SQL基础知识，包括从多个表组合数据的方法。
- [Database System Concepts](https://www.mheducation.com/highered/product/database-system-concepts-silberschatz-korth/M9780078022159.html) — Abraham Silberschatz, Henry Korth, S. Sudarshan (2020)
  Publisher: McGraw-Hill Education
  一本广泛使用的学术教材，涵盖了数据库系统、关系模型和SQL查询处理的理论基础。
