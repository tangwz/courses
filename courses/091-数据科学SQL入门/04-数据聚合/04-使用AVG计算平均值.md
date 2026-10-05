# 使用AVG计算平均值

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/calculating-averages-avg)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据分析中，一项常见需求是找出数值列的平均值。无论你需要订单总额的平均值、客户满意度的平均分数，还是典型的会话时长，`AVG()`函数都是完成这项工作的工具。它计算一组数字的算术平均值。

### `AVG()`的基本用法

`AVG()`函数通过将指定列中所有非空数值求和，然后除以这些非空值的计数来工作。

基本语法很简单：

```sql
SELECT AVG(column_name)
FROM table_name;
```

我们来看一个包含客户购买信息的`orders`表：

| 订单ID | 客户ID | 订单日期 | 订单总额 |
| --- | --- | --- | --- |
| 101 | 5 | 2023-10-26 | 150.75 |
| 102 | 12 | 2023-10-26 | 85.00 |
| 103 | 5 | 2023-10-27 | 210.50 |
| 104 | 21 | 2023-10-27 | NULL |
| 105 | 12 | 2023-10-28 | 110.25 |

为了找出所有已完成订单的平均`order_total` (忽略总额为`NULL`的订单)，你可以这样写：

```sql
SELECT AVG(order_total)
FROM orders;
```

数据库执行此计算：$(150.75 + 85.00 + 210.50 + 110.25) / 4 = 556.50 / 4 = 139.125$。

结果可能如下所示：

| 平均值 |
| --- |
| 139.125000 |

小数位的确切数量可能不同，这取决于数据库系统和`order_total`列的数据类型。

### `AVG()`中别名的使用

与其他函数和列一样，默认的输出列名 (通常是`avg`或类似名称) 可能不够有描述性。你可以使用`AS`关键字通过别名指定一个更有意义的名称：

```sql
SELECT AVG(order_total) AS average_order_value
FROM orders;
```

这个查询会产生一个更清晰的结果：

| 平均订单价值 |
| --- |
| 139.125000 |

### 使用`WHERE`计算子集的平均值

通常，你只会想计算数据中特定部分的平均值。你可以将`AVG()`与`WHERE`子句结合使用，在计算平均值*之前*筛选行。

例如，要找出`customer_id`为12的客户的平均订单总额：

```sql
SELECT AVG(order_total) AS avg_order_total_cust12
FROM orders
WHERE customer_id = 12;
```

此查询首先筛选`orders`表，以仅包含`customer_id`为12的行：

| 订单ID | 客户ID | 订单日期 | 订单总额 |
| --- | --- | --- | --- |
| 102 | 12 | 2023-10-26 | 85.00 |
| 105 | 12 | 2023-10-28 | 110.25 |

然后，它计算这些行的平均`order_total`：$(85.00 + 110.25) / 2 = 195.25 / 2 = 97.625$。

结果是：

| 客户12的平均订单总额 |
| --- |
| 97.625000 |

### 关于`NULL`值的重要说明

记住`AVG()`这一点很重要，与大多数聚合函数一样 (除了`COUNT(*)`), 它完全忽略`NULL`值。它计算非`NULL`值的总和，并除以非`NULL`值的*计数*。在我们最初的例子中，`order_id`为104的行被排除在求和和计数之外。

请注意这种行为。如果`NULL`在你的数据环境中实际上表示零，那么简单地使用`AVG()`会给出非零值的平均值，这可能不是你想要的。如果你需要在平均值计算中将`NULL`视为零，你通常需要使用一个函数在聚合*之前*将`NULL`替换为0，但这是一种更高级的技巧。目前，只需记住`AVG()`计算的是*现有*值的平均值。

### 数据类型考量

`AVG()`函数是为数值数据类型设计的，例如`INTEGER`、`DECIMAL`、`FLOAT`、`NUMERIC`等。尝试在非数值列 (如文本或日期) 上使用`AVG()`通常会导致错误。

计算平均值是数据汇总的一个基本步骤。`AVG()`提供了一种直接方法来计算针对整个列或由`WHERE`子句定义的特定子集的平均值。稍后，你将了解如何使用`GROUP BY`子句同时计算多个组的平均值。

## 参考资料

- [PostgreSQL 16 Documentation: Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — PostgreSQL Global Development Group (2025)
  Publisher: PostgreSQL Global Development Group
  PostgreSQL官方文档，描述包括AVG()在内的聚合函数的行为和语法。
- [MySQL 8.0 Reference Manual: Aggregate (Group-By) Functions](https://dev.mysql.com/doc/refman/8.0/en/group-by-functions.html) — Oracle and/or its affiliates (2024)
  Publisher: Oracle
  MySQL官方文档，详细介绍了聚合函数及其用法，与AVG()相关。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057631/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本关于SQL基础的实用指南，其中一个章节专门讨论了AVG()等聚合函数。第三版。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20SUM%20%E8%AE%A1%E7%AE%97%E6%80%BB%E5%92%8C.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20MIN-MAX%20%E5%87%BD%E6%95%B0%E6%9F%A5%E6%89%BE%E6%9C%80%E5%B0%8F%E5%80%BC%E5%92%8C%E6%9C%80%E5%A4%A7%E5%80%BC.md)
