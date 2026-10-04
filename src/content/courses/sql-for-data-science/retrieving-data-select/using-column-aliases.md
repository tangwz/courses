---
course: "sql-for-data-science"
chapter: "retrieving-data-select"
lesson: "using-column-aliases"
sourceId: 1567
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-2-retrieving-data-select/using-column-aliases"
title: "使用列别名"
description: "通过为列名分配别名来提高查询结果的可读性。"
order: 4
plots: []
sourceHash: "1fa2fd19db9d99b6117076e74ff3b13b7a1e0cffd0c10ae879eeb97ecce03207"
sourceCorrections: []
---

`SELECT` 语句允许从表中选择所需的列。有时，数据库中的原始列名不适合最终报告或数据处理。它们可能缩写（例如 `prod_id`）、技术性强（例如 `calculated_value_stream_1`），或者对于受众来说描述性不足。此外，当在 `SELECT` 语句中直接进行计算时（这将在之后讲到），数据库通常会为计算所得的列分配一个通用或无用的名称。

此时，列别名就派上用场了。别名是您在一个特定查询范围内为列分配的临时替代名称。它不会改变数据库表中实际的列名；它只影响列标题在该特定 `SELECT` 语句结果中的显示方式。使用别名能大幅提升查询结果的可读性和清晰度。

### 使用 `AS` 分配别名

为列分配别名的标准方式是使用 `AS` 关键字，后跟所需的别名。

假设我们有一个名为 `Products` 的表，其中包含 `ProductID`、`ProductName` 和 `UnitPrice` 列。如果想获取产品名称和单价，但以更友好的标题如“产品名称”和“价格”来显示，我们可以这样写：

```sql
SELECT
  ProductName AS "Product Name",
  UnitPrice AS "Price"
FROM Products;
```

**说明：**

1. `SELECT ProductName AS "Product Name"`: 这会选择 `ProductName` 列，但告知数据库在结果集中以“产品名称”作为标题显示它。
2. `UnitPrice AS "Price"`: 类似地，这会选择 `UnitPrice` 列并将其别名为“价格”。
3. `FROM Products`: 指定我们正在查询的表。

**结果：**

结果将不再显示 `ProductName` 和 `UnitPrice` 作为列标题，而是像这样：

| 产品名称 | 价格 |
| --- | --- |
| Laptop Pro | 1200 |
| Wireless Mouse | 25 |
| External Hard Drive | 80 |
| ... | ... |

**别名中使用引号：**

请注意“产品名称”周围使用的双引号 (`"`)。如果您的别名：

- 包含空格（如“产品名称”）
- 包含特殊字符（如“价格 (\$)”）
- 是 SQL 中的保留关键字（如 `SELECT` 或 `ORDER`，尽管通常不推荐这样做）
- 需要保持特定的字母大小写（某些数据库系统对别名区分大小写）

您*必须*将别名用双引号 (`"`) 括起来。一些数据库系统也可能为此目的使用单引号 (`'`) 或方括号 (`[]`)，但双引号是最常见的标准。如果您的别名是一个没有空格或特殊字符的单词（如 `Price`），则引号通常是可选的。然而，一致地使用引号，特别是对于多词别名，是提高清晰度和避免潜在错误的好习惯。

### 隐式别名（可选 `AS`）

许多 SQL 数据库系统也允许您通过将所需别名直接放在列名之后，并用空格隔开，而不必明确使用 `AS` 关键字来分配别名。

使用之前的例子：

```sql
SELECT
  ProductName "Product Name", -- 此处省略了 AS
  UnitPrice Price           -- 此处省略了 AS，单字别名引号也可选
FROM Products;
```

这个查询产生的结果与使用 `AS` 的查询完全相同。虽然这种简写可以节省一些输入，但明确使用 `AS` 通常使查询更易于阅读和理解，特别是对于复杂的查询或当其他人需要检查您的代码时。对于初学者，在您对阅读 SQL 非常熟悉之前，通常建议坚持使用 `AS`。

### 为什么要使用别名？

- **可读性：** 为输出中的列提供清晰、有意义的名称，使报告和结果更容易理解。
- **计算的清晰性：** 在执行计算时（例如 `UnitPrice * Quantity`），别名为计算所得的值提供一个名称。
- **处理重复名称：** 如果您连接的表（之后会讲到）具有同名列（例如，多个表中的 `id` 列），别名在结果集中区分它们非常重要。
- **简洁性：** 缩短长或复杂的列名，以便在查询本身中更易于引用（在使用表别名时特别有用，这将在关于 JOIN 的章节中讨论）。

通过使用别名即时重命名列，您可以在不更改底层数据库结构的情况下，对查询结果的呈现和可用性获得重要的控制。

## 参考资料

- [PostgreSQL 16.2 Documentation: 7.3. Column Labels (Aliases)](https://www.postgresql.org/docs/current/queries-select-lists.html#QUERIES-COLUMN-LABELS) — The PostgreSQL Global Development Group (2024)
  PostgreSQL的官方文档，解释了SELECT语句中的列别名（标签），包括AS关键字的使用和隐式别名。
- [Learning SQL, Third Edition](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057631/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本SQL基础知识的指南，其中包含一个关于列别名的具体章节，用于提高查询输出的可读性。
