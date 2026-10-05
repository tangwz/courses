# 使用 INNER JOIN

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/using-inner-join)

[返回章节目录](README.md) · [返回课程目录](../README.md)

主键和外键在表之间创建关联，从而可以组合这些关联表中的数据。在 SQL 中，最常用且最基本的方法是使用 `INNER JOIN` 子句。

可以把 `INNER JOIN` 看作一种根据公共列水平合并两个表的方法。它检查您指定的两个表中的行，并且只选择那些在指定“连接列”中值完全匹配的行对。如果一个表中的行在连接条件中没有在另一个表中找到对应的匹配项，那么它将不会出现在最终结果集中。

### `INNER JOIN` 语法

使用 `INNER JOIN` 的基本结构如下：

```sql
SELECT table1.column1, table1.column2, table2.columnA, table2.columnB
FROM table1
INNER JOIN table2
ON table1.common_column = table2.common_column;
```

让我们分解一下这些组成部分：

1. **`SELECT table1.column1, table2.columnA ...`**：您指定希望在最终结果中看到的列。请注意，当两张表中有同名列（例如常见的 ID 列）时，或仅仅为了清晰起见，您需要用表名作为列名的前缀（`table1.column` 或 `table2.column`）。稍后我们将看到一种使用别名缩短表名的方法。
2. **`FROM table1`**：这表示连接中的第一个表（通常被称为“左”表）。
3. **`INNER JOIN table2`**：这指定了您要连接的第二个表（“右”表）以及连接类型（`INNER`）。
4. **`ON table1.common_column = table2.common_column`**：这是连接条件。它是最重要的部分，它告诉数据库这些表是如何关联的。您指定 `table1` 中的哪个列应该与 `table2` 中的哪个列匹配。通常，这涉及将一个表的主键与另一个表中的对应外键进行匹配。

### 示例：客户和订单

假设我们的数据库中有两个表：`Customers`（客户）和 `Orders`（订单）。

**`Customers`（客户）表：**

| 客户ID | 名 | 姓 | 城市 |
| --- | --- | --- | --- |
| 1 | Clara | Oswald | London |
| 2 | Amy | Pond | Leadworth |
| 3 | Donna | Noble | London |
| 4 | Martha | Jones | London |

**`Orders`（订单）表：**

| 订单ID | 产品 | 数量 | 客户ID |
| --- | --- | --- | --- |
| 101 | Screwdriver | 1 | 2 |
| 102 | TARDIS Key | 1 | 1 |
| 103 | Celery | 5 | 5 |
| 104 | Scarf | 1 | 2 |

`Customers` 表使用 `customer_id` 作为其主键。 `Orders` 表使用 `customer_id` 作为外键关联到 `Customers` 表。

现在，假设我们想查看订单列表，并显示下单客户的姓名。我们需要来自两个表的信息。我们可以使用 `INNER JOIN` 来实现此目的：

```sql
SELECT
    O.order_id,
    O.product,
    C.first_name,
    C.last_name
FROM
    Orders AS O
INNER JOIN
    Customers AS C
ON
    O.customer_id = C.customer_id;
```

- 我们从 `Orders` 表中选择 `order_id` 和 `product`（使用 `O` 作为 `Orders` 的简称别名）。
- 我们从 `Customers` 表中选择 `first_name` 和 `last_name`（使用 `C` 作为 `Customers` 的别名）。
- 我们从 `Orders` 表开始。
- 将其与 `Customers` 表进行 `INNER JOIN`。
- 重要的 `ON` 子句指定了表之间的关联条件，即 `Orders` 表中的 `customer_id`（`O.customer_id`）与 `Customers` 表中的 `customer_id`（`C.customer_id`）匹配。

### 理解结果

运行此查询将产生以下结果：

| 订单ID | 产品 | 名 | 姓 |
| --- | --- | --- | --- |
| 101 | Screwdriver | Amy | Pond |
| 102 | TARDIS Key | Clara | Oswald |
| 104 | Scarf | Amy | Pond |

请注意以下几点：

1. **包含匹配的行**：订单 101 关联到客户 2 (Amy Pond)，订单 102 关联到客户 1 (Clara Oswald)，订单 104 再次关联到客户 2 (Amy Pond)。所有这些都显示出来，因为在 `Orders` 和 `Customers` 表中都找到了匹配的 `customer_id`。
2. **排除不匹配的行**：
   - Donna Noble（客户 3）和 Martha Jones（客户 4）不在结果中，因为他们在 `Orders` 表中没有对应的订单。
   - 订单 103 不在结果中，因为其 `customer_id` (5) 在 `Customers` 表中不存在。

这种排除不匹配行的特性是 `INNER JOIN` 的显著特点。它只返回根据 `ON` 条件存在关联的行。

> 此图演示了 `INNER JOIN`。线条连接了 `Customers` 和 `Orders` 表中 `customer_id` 匹配的行。只有这些关联的行组成了最终的结果集。客户 3、客户 4 和订单 103 被排除在外，因为它们在另一个表中没有匹配项。

`INNER JOIN` 是组合关联数据的重要工具。熟练掌握其语法并理解它如何根据 `ON` 条件匹配行，对于有效查询关系型数据库非常重要。在接下来的部分中，我们将介绍如何连接两个以上的表以及如何使用别名使查询更简洁。

## 参考资料

- [7.2. Join Tables](https://www.postgresql.org/docs/17/tutorial-joins.html) — PostgreSQL Global Development Group (2025)
  Publisher: PostgreSQL Global Development Group
  PostgreSQL的官方文档，清晰解释了SQL JOIN子句，包括INNER JOIN的语法、行为和示例。
- [Learning SQL, 3rd Edition](https://www.oreilly.com/library/view/learning-sql-3rd/9781492070851/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: Chapter 9: Joining Tables
  一本SQL基础知识的实用指南，其中有一个专门章节详细介绍了各种连接类型的使用，包括对INNER JOIN的详尽阐述。
- [Database System Concepts, 7th Edition](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill; Pages: Chapter 3: Introduction to SQL
  一本广泛使用的数据库系统学术教材，在关系数据库理论和实际应用的背景下解释了SQL连接操作。

---

[上一节](03-SQL%20JOINs%20%E7%AE%80%E4%BB%8B.md) · [下一节](05-%E8%BF%9E%E6%8E%A5%E5%A4%9A%E4%B8%AA%E8%A1%A8.md)
