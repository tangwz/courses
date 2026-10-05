# 使用 SUM 计算总和

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/calculating-sums-sum)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 `SUM()` 聚合函数计算总和是一种汇总数据的基本方法。通常，您会需要知道多行数据的总值，例如总销售收入、商品销售总量或总行驶距离。`SUM()` 函数专门为此目的设计。

## `SUM()` 函数的作用

`SUM()` 函数计算指定数值列中所有值的总和。它作用于一组行（可以是整个表，也可以是由 `WHERE` 子句定义的子集），并返回一个代表总和的单一值。

需要记住的是，`SUM()` 只适用于 `INTEGER`、`DECIMAL`、`FLOAT`、`NUMERIC` 等数值数据类型。尝试对文本 (`VARCHAR`) 或日期 (`DATE`) 列使用 `SUM()` 会导致数据库报错。

## 基本语法

使用 `SUM()` 的基本结构简单明了：

```sql
SELECT SUM(numeric_column)
FROM table_name;
```

这里，`numeric_column` 是您想要求和的列名，`table_name` 是包含该列的表名。

## 计算总量

假设我们有一个 `OrderItems` 表，记录客户订单中包含的每个商品：

**`OrderItems` 表：**

| OrderItemID | OrderID | ProductID | Quantity | UnitPrice |
| --- | --- | --- | --- | --- |
| 1 | 101 | P001 | 2 | 10.00 |
| 2 | 101 | P005 | 1 | 25.50 |
| 3 | 102 | P002 | 5 | 5.00 |
| 4 | 103 | P001 | 3 | 10.00 |
| 5 | 103 | P008 | 1 | 150.00 |
| 6 | 104 | P005 | 4 | 25.50 |

为了找出所有订单中售出商品的总数量，我们可以对 `Quantity` 列求和：

```sql
SELECT SUM(Quantity)
FROM OrderItems;
```

此查询处理 `OrderItems` 表中每一行的 `Quantity` 列，并返回一个单一结果：

| SUM(Quantity) |
| --- |
| 16 |

结果 16 是 2 + 1 + 5 + 3 + 1 + 4 的总和。

## 计算总收入

我们也可以在 `SUM()` 函数内部进行计算。例如，要计算所有订单商品的总收入，我们需要将每个商品的 `Quantity` 乘以 `UnitPrice`，然后将这些结果相加。

```sql
SELECT SUM(Quantity * UnitPrice)
FROM OrderItems;
```

数据库首先计算每一行的 `Quantity * UnitPrice`：

- 行 1: $2 * 10.00 = 20.00$
- 行 2: $1 * 25.50 = 25.50$
- 行 3: $5 * 5.00 = 25.00$
- 行 4: $3 * 10.00 = 30.00$
- 行 5: $1 * 150.00 = 150.00$
- 行 6: $4 * 25.50 = 102.00$

然后，它将这些值相加：$20.00 + 25.50 + 25.00 + 30.00 + 150.00 + 102.00 = 352.50$。

查询结果会是：

| SUM(Quantity \* UnitPrice) |
| --- |
| 352.50 |

## 使用别名提高清晰度

结果的默认列名（例如 `SUM(Quantity)` 或 `SUM(Quantity * UnitPrice)`）描述性不强。正如我们在选择列时所看到的，我们可以使用 `AS` 关键字为 `SUM()` 计算结果赋予一个更有意义的别名。

```sql
SELECT
  SUM(Quantity) AS TotalItemsSold,
  SUM(Quantity * UnitPrice) AS TotalRevenue
FROM OrderItems;
```

此查询一次性计算出两个总和，并以清晰的列名展示：

| TotalItemsSold | TotalRevenue |
| --- | --- |
| 16 | 352.50 |

## 使用 WHERE 对过滤后的数据求和

就像 `COUNT()` 一样，`SUM()` 函数可以与 `WHERE` 子句结合使用，以便仅计算您数据特定子集的总和。例如，我们来计算仅涉及 `ProductID` 为 'P001' 的订单所产生的总收入。

```sql
SELECT SUM(Quantity * UnitPrice) AS RevenueFromP001
FROM OrderItems
WHERE ProductID = 'P001';
```

数据库首先过滤 `OrderItems` 表，只包含 `ProductID` 为 'P001' 的行（我们例子中的第 1 行和第 4 行）。然后，它对这些过滤后的行计算 `Quantity * UnitPrice` 并应用 `SUM()` 函数：

- 行 1: $2 * 10.00 = 20.00$
- 行 4: $3 * 10.00 = 30.00$
- 总和: $20.00 + 30.00 = 50.00$

结果：

| RevenueFromP001 |
| --- |
| 50.00 |

## `SUM()` 如何处理 NULL 值

如果您求和的列包含 `NULL` 值会发生什么？ `SUM()` 函数会直接忽略它们。它仅使用列中非 `NULL` 值进行求和。如果所选行中该列的所有值都是 `NULL`，`SUM()` 会返回 `NULL`。这种行为通常是您想要的，因为 `NULL` 通常代表未知或缺失数据，在求和时不应被视为零。

例如，如果 `OrderItemID` 为 5 的 `UnitPrice` 是 `NULL`，那么 `TotalRevenue` 计算将完全排除该行的贡献。

`SUM()` 函数是计算数据集中总和的强大工具，无论是针对整个表还是特定子集。接下来，我们将了解如何使用 `AVG()` 函数计算平均值。

## 参考资料

- [PostgreSQL: Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — PostgreSQL Global Development Group (2023)
  PostgreSQL聚合函数的官方文档，解释了SUM()的行为、语法和空值处理。
- [MySQL: Aggregate Functions](https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html) — Oracle and/or its affiliates (2023)
  Publisher: Oracle
  MySQL聚合函数的官方文档，涵盖了SUM()的使用、数值类型和结果管理。
- [Learning SQL: Generate, Manipulate, and Retrieve Data](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本备受推崇的SQL入门书籍，其中包含关于SUM()等聚合函数的详细章节和实际示例。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20COUNT%20%E7%BB%9F%E8%AE%A1%E8%A1%8C%E6%95%B0.md) · [下一节](04-%E4%BD%BF%E7%94%A8AVG%E8%AE%A1%E7%AE%97%E5%B9%B3%E5%9D%87%E5%80%BC.md)
