---
course: "sql-for-data-science"
chapter: "aggregating-data"
lesson: "grouping-data-groupby"
sourceId: 1613
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/grouping-data-groupby"
title: "用 `GROUP BY` 进行数据分组"
description: "学习如何使用 `GROUP BY` 根据列值将聚合函数应用于行的子集。"
order: 6
plots: []
sourceHash: "266d46a0a5f66448e1a6923353356b8bb0e25242819cd6c4fa00eb066b44860a"
sourceCorrections: []
---

聚合函数，如 `COUNT()`、`SUM()` 和 `AVG()`，用于计算查询返回的所有行的汇总统计信息。虽然获得一个单一的数字（例如总订单数或整个产品目录的平均产品价格）很有用，但通常你需要更细致的汇总。例如，你可能不想要整体的平均产品价格，而是想知道*每个产品类别*的平均价格。或者，你可能不想要客户总数，而是想知道*每个城市中*的客户数量。

这就是 `GROUP BY` 子句发挥作用的地方。它允许你根据一个或多个列中的值，将表的行分成更小的组。然后，聚合函数会*独立地应用于这些组中的每一个*。

### `GROUP BY` 的工作机制

可以把 `GROUP BY` 看作是在聚合发生之前重新组织数据的过程：

1. **分区：** SQL 逻辑上将 `FROM` 和 `WHERE` 子句返回的行划分为组。组内的所有行在 `GROUP BY` 子句中指定的列中具有*相同的值*。
2. **聚合：** `SELECT` 子句中列出的聚合函数（例如 `COUNT()`、`AVG()`、`SUM()`、`MIN()`、`MAX()`）随后分别应用于每个组。
3. **结果：** 查询为每个组返回一行汇总结果。

基本语法如下：

```sql
SELECT
    column_to_group_by,
    aggregate_function(column_to_aggregate)
FROM
    table_name
WHERE
    -- 可选：在分组之前过滤行
    condition
GROUP BY
    column_to_group_by
ORDER BY
    -- 可选：对分组结果进行排序
    column_to_group_by; -- 或聚合函数结果
```

注意顺序：`GROUP BY` 在 `FROM` 和 `WHERE` 之后，但在 `ORDER BY` 之前。

### 简单分组示例

让我们使用一个包含客户订单信息的 `Orders` 表：

| order\_id | customer\_id | order\_date | order\_total |
| --- | --- | --- | --- |
| 101 | 1 | 2023-10-01 | 50.00 |
| 102 | 2 | 2023-10-01 | 120.50 |
| 103 | 1 | 2023-10-05 | 75.25 |
| 104 | 3 | 2023-10-06 | 30.00 |
| 105 | 2 | 2023-10-08 | 80.00 |
| 106 | 1 | 2023-10-10 | 45.75 |

假设我们想找到每个客户的总花费金额。我们需要按 `customer_id` 对行进行分组，然后对每个组的 `order_total` 应用 `SUM()` 函数。

```sql
SELECT
    customer_id,
    SUM(order_total) AS total_spent
FROM
    Orders
GROUP BY
    customer_id;
```

SQL 的处理过程如下：

1. 它查看 `Orders` 表。
2. 它根据 `customer_id` 对行进行分组：
   - 组 1：`customer_id` = 1（订单 101、103、106）
   - 组 2：`customer_id` = 2（订单 102、105）
   - 组 3：`customer_id` = 3（订单 104）
3. 它为每个组计算 `SUM(order_total)`：
   - 组 1：$50.00 + 75.25 + 45.75 = 171.00$
   - 组 2：$120.50 + 80.00 = 200.50$
   - 组 3：$30.00 = 30.00$
4. 它为每个组返回一行：

| customer\_id | total\_spent |
| --- | --- |
| 1 | 171.00 |
| 2 | 200.50 |
| 3 | 30.00 |

我们还可以使用 `COUNT(*)` 找到每个客户的订单数量：

```sql
SELECT
    customer_id,
    COUNT(*) AS number_of_orders
FROM
    Orders
GROUP BY
    customer_id;
```

这将返回：

| customer\_id | number\_of\_orders |
| --- | --- |
| 1 | 3 |
| 2 | 2 |
| 3 | 1 |

### `GROUP BY` 的 `SELECT` 列表规则

使用 `GROUP BY` 时有一条重要规则：**`SELECT` 列表中任何*不是*聚合函数的列都必须包含在 `GROUP BY` 子句中。**

为什么？考虑查询 `SELECT customer_id, SUM(order_total) FROM Orders GROUP BY customer_id;`。这之所以可行，是因为对于每个组（由单个 `customer_id` 定义），都有一个单一的 `customer_id` 值和一个单一的 `SUM(order_total)` 结果。

现在，假设你尝试这样做：

```sql
-- 此查询在标准 SQL 中通常无效
SELECT
    customer_id,
    order_date, -- 未聚合，也不在 GROUP BY 中
    SUM(order_total) AS total_spent
FROM
    Orders
GROUP BY
    customer_id;
```

对于 `customer_id` 为 1 的情况，存在多个 `order_date` 值（'2023-10-01'、'2023-10-05'、'2023-10-10'）。由于查询将这三行合并为 `customer_id` 1 的单个输出行，SQL 不知道应该显示*哪个* `order_date`。为避免这种模糊性，SQL 要求 `SELECT` 列表中任何非聚合列也必须包含在 `GROUP BY` 子句中。如果你将 `order_date` 放在 `GROUP BY` 中，你将根据 `customer_id` 和 `order_date` 的组合进行分组，这是一种不同的计算。

### 按多个列分组

你不限于只按一个列进行分组。你可以在 `GROUP BY` 子句中指定多个列，根据这些列中值的独特*组合*来创建更细致的组。

让我们在 `Orders` 表中添加一个 `product_category` 列（为简化起见，假设每个订单只属于一个类别）：

| order\_id | customer\_id | product\_category | order\_total |
| --- | --- | --- | --- |
| 101 | 1 | Books | 50.00 |
| 102 | 2 | Electronics | 120.50 |
| 103 | 1 | Groceries | 75.25 |
| 104 | 3 | Books | 30.00 |
| 105 | 2 | Groceries | 80.00 |
| 106 | 1 | Books | 45.75 |

现在，让我们找到每个客户*在每个产品类别内*的总花费：

```sql
SELECT
    customer_id,
    product_category,
    SUM(order_total) AS category_total_spent
FROM
    Orders
GROUP BY
    customer_id, product_category -- 按组合分组
ORDER BY
    customer_id, product_category; -- 可选排序
```

SQL 现在根据唯一的 `(customer_id, product_category)` 对形成组：

- (1, Books) -> 订单 101, 106 -> $50.00 + 45.75 = 95.75$
- (1, Groceries) -> 订单 103 -> $75.25$
- (2, Electronics) -> 订单 102 -> $120.50$
- (2, Groceries) -> 订单 105 -> $80.00$
- (3, Books) -> 订单 104 -> $30.00$

结果将是：

| customer\_id | product\_category | category\_total\_spent |
| --- | --- | --- |
| 1 | Books | 95.75 |
| 1 | Groceries | 75.25 |
| 2 | Electronics | 120.50 |
| 2 | Groceries | 80.00 |
| 3 | Books | 30.00 |

该图演示了 `GROUP BY customer_id, product_category` 如何根据客户和类别的独特组合划分原始行，从而可以为每个不同的对计算 `SUM(order_total)`。

### 与 `WHERE` 的交互

请记住，`WHERE` 子句在 `GROUP BY` 操作发生*之前*过滤单行。如果你只想针对原始数据的一个子集计算聚合，请使用 `WHERE`。例如，要找到每个客户的总花费，但只考虑在 '2023-10-04' 之后下的订单：

```sql
SELECT
    customer_id,
    SUM(order_total) AS total_spent_recent
FROM
    Orders
WHERE
    order_date > '2023-10-04' -- 首先过滤行
GROUP BY
    customer_id;          -- 对剩余行进行分组
```

此查询首先移除订单 101 和 102，*然后*在对总额求和之前，按 `customer_id` 对剩余的行（103、104、105、106）进行分组。

`GROUP BY` 子句是从数据中创建有意义汇总的基础。它与聚合函数协同工作，将多行合并为针对数据集中不同组的有信息的汇总统计信息。接下来，我们将研究如何使用 `HAVING` 子句过滤这些*分组*后的结果。

## 参考资料

- [SQL SELECT Statement (including GROUP BY)](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2023)
  官方文档，解释 SELECT 语句的结构，包括 GROUP BY 子句及其数据聚合规则。
