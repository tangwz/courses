---
course: "sql-for-data-science"
chapter: "aggregating-data"
lesson: "practice-aggregation-grouping"
sourceId: 1620
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/practice-aggregation-grouping"
title: "动手实践：聚合与分组"
description: "编写使用聚合函数、GROUP BY 和 HAVING 来汇总数据的查询。"
order: 8
plots: ["plots/1620-0.json"]
sourceHash: "d6699ccabab015d121319658ce3fbc6145880754bd92eb5a91e8859bcff07d43"
sourceCorrections: []
---

聚合函数，包括 `COUNT`、`SUM`、`AVG`、`MIN` 和 `MAX`，用于汇总数据。`GROUP BY` 用于在特定的数据子集上执行这些计算，而 `HAVING` 则根据这些聚合结果进行筛选。

对于这些练习，假设我们有两个表：`Products` 和 `Orders`。

**示例 `Products` 表：**

| product\_id | product\_name | category | price |
| --- | --- | --- | --- |
| 101 | Quantum Widget | Electronics | 75.00 |
| 102 | Flux Capacitor | Electronics | 1200.00 |
| 103 | Turbo Encabulator | Gadgets | 250.50 |
| 104 | Sky Hook | Hardware | 35.20 |
| 105 | Left-Handed Wrench | Hardware | 15.00 |
| 106 | Photon Phaser | Gadgets | 150.75 |
| 107 | Sonic Screwdriver | Gadgets | 99.99 |

**示例 `Orders` 表：**

| order\_id | customer\_id | order\_date | order\_total |
| --- | --- | --- | --- |
| 1 | CUST-A | 2023-10-26 | 150.75 |
| 2 | CUST-B | 2023-10-26 | 75.00 |
| 3 | CUST-A | 2023-10-27 | 250.50 |
| 4 | CUST-C | 2023-10-28 | 50.20 |
| 5 | CUST-B | 2023-10-28 | 1275.00 |
| 6 | CUST-A | 2023-10-29 | 35.20 |

让我们开始吧。

### 计算总体统计数据

首先，让我们对订单有一个总体了解。我们希望找到订单总数、所有订单的总销售额、平均订单价值以及最高和最低订单金额。

```sql
SELECT
    COUNT(*) AS total_orders,
    SUM(order_total) AS total_revenue,
    AVG(order_total) AS average_order_value,
    MIN(order_total) AS minimum_order_value,
    MAX(order_total) AS maximum_order_value
FROM
    Orders;
```

**解释：**

- `COUNT(*)`: 计算 `Orders` 表中的总行数。我们使用别名 `total_orders` 使输出清晰。
- `SUM(order_total)`: 计算 `order_total` 列中所有值的总和，得到总收入。别名为 `total_revenue`。
- `AVG(order_total)`: 计算 `order_total` 列的平均值。别名为 `average_order_value`。请记住，这是通过 $SUM(order\_total) / COUNT(order\_total)$ 计算得出的。
- `MIN(order_total)`: 找到 `order_total` 列中的最小值。别名为 `minimum_order_value`。
- `MAX(order_total)`: 找到 `order_total` 列中的最大值。别名为 `maximum_order_value`。

**预期结果：**

（基于示例数据）

| 订单总数 | 总收入 | 平均订单价值 | 最低订单价值 | 最高订单价值 |
| --- | --- | --- | --- | --- |
| 6 | 1836.65 | 306.1083 | 35.20 | 1275.00 |
| *注意：具体的平均值可能会根据 SQL 数据库系统的不同而显示更多或更少的小数位数。* |  |  |  |  |

### 分组结果：按客户统计订单

现在，让我们看看每个客户下了多少订单。这需要先按 `customer_id` 分组行，然后再计数。

```sql
SELECT
    customer_id,
    COUNT(*) AS number_of_orders
FROM
    Orders
GROUP BY
    customer_id;
```

**解释：**

- `GROUP BY customer_id`: 此子句根据 `customer_id` 列中的唯一值对 `Orders` 表中的行进行分组。所有“CUST-A”的行都在一个组中，所有“CUST-B”的行在另一个组中，依此类推。
- `COUNT(*)`: 此聚合函数现在分别应用于*每个组*。它计算每个客户组内的行数。
- `SELECT customer_id, ...`: 我们选择 `customer_id` 来了解每个计数属于哪个组。

**预期结果：**

| 客户ID | 订单数量 |
| --- | --- |
| CUST-A | 3 |
| CUST-B | 2 |
| CUST-C | 1 |

### 多聚合分组：客户消费习惯

让我们在前一个查询的基础上进行扩展。对于每个客户，我们不仅要找出他们下了多少订单，还要找出他们的总消费额和平均订单价值。

```sql
SELECT
    customer_id,
    COUNT(*) AS number_of_orders,
    SUM(order_total) AS total_spent,
    AVG(order_total) AS average_spent
FROM
    Orders
GROUP BY
    customer_id;
```

**解释：**

- 此查询与上一个查询类似，仍然按 `customer_id` 分组。
- 我们添加了 `SUM(order_total)` 和 `AVG(order_total)`。这些函数是针对 `GROUP BY` 子句创建的每个客户组独立计算的。

**预期结果：**

| 客户ID | 订单数量 | 总消费额 | 平均消费额 |
| --- | --- | --- | --- |
| CUST-A | 3 | 436.45 | 145.4833 |
| CUST-B | 2 | 1350.00 | 675.0000 |
| CUST-C | 1 | 50.20 | 50.2000 |

### 按产品类别分组

让我们切换到 `Products` 表。我们可以按 `category` 分组，以了解每个类别中有多少产品以及它们的平均价格是多少。

```sql
SELECT
    category,
    COUNT(*) AS number_of_products,
    AVG(price) AS average_price
FROM
    Products
GROUP BY
    category;
```

**解释：**

- `GROUP BY category`: 根据产品类别（'Electronics'、'Gadgets'、'Hardware'）对行进行分组。
- `COUNT(*)`: 计算每个类别组内的产品数量。
- `AVG(price)`: 计算每个类别组内产品的平均价格。

**预期结果：**

| 类别 | 产品数量 | 平均价格 |
| --- | --- | --- |
| Electronics | 2 | 637.5000 |
| Gadgets | 3 | 167.0800 |
| Hardware | 2 | 25.1000 |

让我们将每个类别的平均价格可视化。



![按类别划分的平均产品价格](plots/1620-0.json)



> 计算了每个不同类别中产品的平均价格。

### 使用 HAVING 筛选组

假设我们只想找出那些“大客户”（这里定义为总消费额超过 500 美元）。我们首先需要计算每个客户的总消费额（按 `customer_id` 分组的 `SUM(order_total)`），然后筛选这些汇总结果。这就是 `HAVING` 发挥作用的地方。

```sql
SELECT
    customer_id,
    SUM(order_total) AS total_spent
FROM
    Orders
GROUP BY
    customer_id
HAVING
    SUM(order_total) > 500;
```

**解释：**

- 查询首先按 `customer_id` 分组订单，并计算每个客户的 `SUM(order_total)`。
- `HAVING SUM(order_total) > 500`: 此子句筛选聚合的*结果*。它只保留计算总和大于 500 的组（客户）。
- 重要的是要记住，不能使用 `WHERE SUM(order_total) > 500`。`WHERE` 子句在聚合发生*之前*筛选单个行。`HAVING` 在聚合发生*之后*筛选组。

**预期结果：**

| 客户ID | 总消费额 |
| --- | --- |
| CUST-B | 1350.00 |

### 结合使用 WHERE、GROUP BY 和 HAVING

让我们尝试一个更复杂的场景，结合行筛选 (`WHERE`)、分组 (`GROUP BY`) 和组筛选 (`HAVING`)。找出每个客户的平均订单总额，但只考虑在 2023-10-26 *之后*下达的订单。此外，只显示平均订单总额（对于这些近期订单）大于 100 美元的客户。

```sql
SELECT
    customer_id,
    AVG(order_total) AS average_recent_order_value
FROM
    Orders
WHERE
    order_date > '2023-10-26' -- 首先筛选行
GROUP BY
    customer_id              -- 对剩余行进行分组
HAVING
    AVG(order_total) > 100;  -- 根据聚合结果筛选组
```

**解释：**

1. `WHERE order_date > '2023-10-26'`: 筛选 `Orders` 表，移除在 2023-10-26 或之前下达的订单。订单 ID 1 和 2 被排除。
2. `GROUP BY customer_id`: 按客户对*剩余*行（订单 3、4、5、6）进行分组。
   - CUST-A 组：订单 3 和 6。
   - CUST-B 组：订单 5。
   - CUST-C 组：订单 4。
3. `AVG(order_total)`: 计算每个组的平均 `order_total`。
   - CUST-A：AVG(250.50, 35.20) = 142.85
   - CUST-B：AVG(1275.00) = 1275.00
   - CUST-C：AVG(50.20) = 50.20
4. `HAVING AVG(order_total) > 100`: 筛选这些计算出的组平均值，只保留大于 100 的。

**预期结果：**

| 客户ID | 近期平均订单价值 |
| --- | --- |
| CUST-A | 142.85 |
| CUST-B | 1275.00 |

本次实践练习展示了如何使用聚合函数、有效分组数据以及根据计算值筛选这些组。在您自己的数据上尝试这些查询，或者使用不同的条件和聚合函数进行修改。持续练习是掌握这些基本 SQL 数据分析操作的最佳途径。

## 参考资料

- [PostgreSQL 16.2 Documentation: 9.20. Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — PostgreSQL Global Development Group (2024)
  解释SQL聚合函数及其用法的官方文档。
- [MySQL 8.0 Reference Manual: 12.20.1 Aggregate (Group) Functions](https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html) — Oracle and/or its affiliates (2024)
  Publisher: Oracle
  MySQL聚合函数的官方参考资料。
- [Learning SQL: Master the Fundamentals of SQL and Relational Databases](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本涵盖SQL基础的书，包括聚合和分组操作。
- [SQL for Data Science (Course 1 in specialization)](https://www.coursera.org/learn/sql-for-data-science) — Sadie St. Lawrence (2023)
  Publisher: Coursera
  一个在线课程，提供SQL数据分析指导，包含聚合和分组部分。
