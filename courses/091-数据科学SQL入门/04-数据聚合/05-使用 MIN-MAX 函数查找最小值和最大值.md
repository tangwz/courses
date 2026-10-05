# 使用 MIN/MAX 函数查找最小值和最大值

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/finding-min-max)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解数据中的值范围通常很有用。你可能需要找到最低价格、最高销售额、最早日期或最新活动。为此，SQL 提供了两个简单但有效的聚合函数：`MIN()` 和 `MAX()`。这些函数有助于确定列中的边界值，无论是针对整个表还是在特定分组内。

### 使用 MIN() 查找最小值

`MIN()` 函数扫描指定列并返回其最小值。这对于确定起始点、最低成本、最早时间或最小数量很有用。

其基本语法很简单：

```sql
SELECT MIN(column_name)
FROM table_name;
```

假设你有一个 `Products` 表，包含有关售出商品的信息，包括它们的价格。要查找可用商品中最便宜的价格，你可以在 `price` 列上使用 `MIN()` 函数：

```sql
SELECT MIN(price) AS lowest_price
FROM Products;
```

此查询查看 `price` 列中的所有值，并只返回找到的单个最小值。我们使用 `AS lowest_price` 来为结果列提供一个描述性名称。

`MIN()` 函数也有效作用于 `GROUP BY` 子句。如果你想查找 *每个* 类别的商品中的最低价格，你可以按商品的 `category` 分组，然后应用 `MIN()` 函数：

```sql
SELECT
  category,
  MIN(price) AS lowest_price_in_category
FROM Products
GROUP BY category;
```

此查询首先依据 `category` 列对行进行分组。然后，对于每个分组（每个不同的类别），它找到 `price` 列中的最小值，并返回类别名称及其对应的最低价格。

### 使用 MAX() 查找最大值

与 `MIN()` 相辅相成，`MAX()` 函数找到指定列中的最大值。这有助于确定峰值、最高成本、最晚时间或最大数量。

语法与 `MIN()` 函数的相似：

```sql
SELECT MAX(column_name)
FROM table_name;
```

再次使用 `Products` 表，如果你想查找最贵的商品，你可以在 `price` 列上使用 `MAX()` 函数：

```sql
SELECT MAX(price) AS highest_price
FROM Products;
```

此查询返回 `price` 列在所有商品中的单个最大值。

和 `MIN()` 类似，`MAX()` 函数也常与 `GROUP BY` 一起使用。要查找每个商品类别中的最高价格：

```sql
SELECT
  category,
  MAX(price) AS highest_price_in_category
FROM Products
GROUP BY category;
```

在这里，查询按 `category` 对商品进行分组，然后确定每个唯一类别的最大 `price` 值。

### 同时使用 MIN() 和 MAX() 函数

你可以在同一个查询中使用 `MIN()` 和 `MAX()`，以快速了解特定属性的值范围。例如，要查找 `Orders` 表中最早和最晚的订单日期：

```sql
SELECT
  MIN(order_date) AS earliest_order_date,
  MAX(order_date) AS latest_order_date
FROM Orders;
```

这提供了订单数据所涵盖时间跨度的简洁概要。

结合 `GROUP BY` 子句，你可以查找每个客户的最低和最高订单总额：

```sql
SELECT
  customer_id,
  MIN(order_total) AS min_order_total,
  MAX(order_total) AS max_order_total
FROM Orders
GROUP BY customer_id;
```

此查询让你了解每个单独客户的消费范围。

### 数据类型和 NULL 处理

`MIN()` 和 `MAX()` 函数如预期工作，适用于数值类型（如 `INTEGER`、`DECIMAL`、`FLOAT`）以及日期/时间类型（如 `DATE`、`TIMESTAMP`）。它们也可以作用于字符串（如 `VARCHAR`、`TEXT`），在这种情况下，它们通常依据字母顺序排序返回“第一个”或“最后一个”值（例如，`MIN()` 返回 'Apple'，`MAX()` 返回 'Zebra'）。

了解一点很重要，`MIN()` 和 `MAX()` 函数，和大多数聚合函数一样，在执行计算时会忽略 `NULL` 值。如果一列只包含 `NULL` 值，那么 `MIN()` 和 `MAX()` 都将返回 `NULL`。

通过使用 `MIN()` 和 `MAX()` 函数，你获得了有价值的工具，以快速评估数据集中的边界和范围，这是数据分析中的一个基本步骤。

## 参考资料

- [PostgreSQL 16.2 Documentation: 9.21. Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — PostgreSQL Global Development Group (2024)
  Publisher: PostgreSQL Global Development Group
  PostgreSQL聚合函数的官方文档，包含MIN和MAX函数、它们的语法以及与GROUP BY和NULL值相关的行为。
- [Learning SQL: Master SQL Fundamentals](https://books.google.com/books?id=i9iV9jYt4-oC) — Alan Beaulieu (2009)
  Publisher: O'Reilly Media
  一本关于SQL基础知识的教科书，提供了聚合函数（包括MIN和MAX）的基础解释。

---

[上一节](04-%E4%BD%BF%E7%94%A8AVG%E8%AE%A1%E7%AE%97%E5%B9%B3%E5%9D%87%E5%80%BC.md) · [下一节](06-%E7%94%A8%20%60GROUP%20BY%60%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%88%86%E7%BB%84.md)
