---
course: "introduction-to-databases"
chapter: "querying-data-sql"
lesson: "select-statement-retrieving-data"
sourceId: 1587
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/select-statement-retrieving-data"
title: "`SELECT` 语句：获取数据"
description: "学习 `SELECT` 语句的基本语法，以从数据库表中获取数据。"
order: 2
plots: []
sourceHash: "357069e3941bbc91100d5deef17bb6f57868ab0a9bcd4bed2b27227992c873b8"
sourceCorrections: []
---

关系型数据库将数据存储在结构化表格中。但是，当你需要数据时，如何实际地将其*取出*呢？SQL中完成这项工作的主要工具是`SELECT`语句。可以将它视为你向数据库询问所包含信息的准确方式。

`SELECT`语句可以说是最常用的SQL命令。它的基本目的是从一个或多个数据库表中获取数据。`SELECT`查询的结果总是一个包含所请求数据的临时表，通常称之为“结果集”。

### 基本语法

`SELECT`语句最简单的形式遵循以下结构：

```sql
SELECT column1, column2, ...
FROM table_name;
```

我们来分析一下：

1. `SELECT`：此关键词告诉数据库你想要获取数据。
2. `column1, column2, ...`：你列出你想要从中获取数据的具体列名，用逗号分隔。
3. `FROM`：此关键词指示数据所在的表格。
4. `table_name`：这是你正在查询的表格的名称。
5. `;`：分号标记 (token)SQL语句的结束。虽然并非所有数据库系统或接口都严格要求，但这是标准做法，有助于分隔多个命令。

### 选择特定列

通常，你只需要表格中少数特定列的数据。为此，你在`SELECT`关键词后列出所需列的名称。

假设我们有一个`Products`表，如下所示：

| ProductID | ProductName | Price | Category |
| --- | --- | --- | --- |
| 1 | Laptop | 1200.00 | Electronics |
| 2 | Mouse | 25.00 | Electronics |
| 3 | Coffee Mug | 15.50 | Home Goods |
| 4 | Notebook | 3.00 | Stationery |

如果你只想查看所有产品的名称和价格，你可以这样写：

```sql
SELECT ProductName, Price
FROM Products;
```

数据库将处理此请求并返回以下结果集：

| ProductName | Price |
| --- | --- |
| Laptop | 1200.00 |
| Mouse | 25.00 |
| Coffee Mug | 15.50 |
| Notebook | 3.00 |

请注意，结果只包含我们明确要求的列（`ProductName`和`Price`），并包含原始`Products`表中的所有行。

### 选择所有列

如果你想查看表格中存储的*所有*信息怎么办？SQL提供了一种方便的简写方式：星号（`*`）。

```sql
SELECT *
FROM Products;
```

`*`作为通配符，表示“所有列”。此查询将返回整个`Products`表，与前面所示完全一致：

| ProductID | ProductName | Price | Category |
| --- | --- | --- | --- |
| 1 | Laptop | 1200.00 | Electronics |
| 2 | Mouse | 25.00 | Electronics |
| 3 | Coffee Mug | 15.50 | Home Goods |
| 4 | Notebook | 3.00 | Stationery |

**何时使用`*`：** 当你首次查看一个表格时，或者在开发过程中需要快速查看所有数据时，使用`SELECT *`非常方便。

**何时避免使用`*`：** 在应用程序或生产代码中，明确列出所需列（`SELECT ProductName, Price ...`）通常是更好的做法。为什么呢？

- **性能：** 获取不必要的列可能会消耗更多资源（网络带宽、内存）。
- **清晰度：** 明确列出列名可以使你的查询意图对其他人（以及未来的自己）更清楚。
- **稳定性：** 如果以后表格中添加或移除了列，`SELECT *`可能会返回意想不到的结果，或者导致依赖特定列集的应用程序代码出错。通过明确列出列，如果表格结构发生变化，但未影响你选择的列，你的查询就不太可能失效。

### 理解结果集

请记住，任何`SELECT`查询的输出本身都是一个类似表格的结构，称为**结果集**。它有列（你选择的那些）和行（符合你条件，在这些基本例子中是所有行）。此结果集通常是临时的；它由数据库生成以响应你的查询，但通常不会永久存储，除非你明确指示它这样做（使用更高级的命令）。

`SELECT`语句是获取信息的基础。你现在已经了解了如何从表格中获取所有列或特定列。接下来，你将学习如何使用`WHERE`子句改进这些查询，以仅获取符合特定条件的行。

## 参考资料

- [Database System Concepts](https://www.mheducation.com/highered/product/database-system-concepts-silberschatz-korth/9781260200257.html) — Abraham Silberschatz, Henry F. Korth, S. Sudarshan (2020)
  Publisher: McGraw-Hill Education
  提供数据库系统的全面学术基础，包含对SQL语法和语义（例如SELECT语句）的详细解释。
- [SELECT](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2023)
  PostgreSQL数据库的官方文档，提供了SELECT语句精确而全面的指南，适用于标准SQL原则。
