# 实践操作：编写 INNER JOIN 查询

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/practice-inner-join)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实践 `INNER JOIN` 查询，组合来自关联表的数据。这些练习将使用一个小型在线商店数据库。

### 数据库结构示例

假设我们有三张表：`Customers`、`Orders` 和 `Products`。

**1. 客户表 (`Customers`)**

此表存储注册客户的信息。

| 客户ID (主键) | 名 | 姓 | 电子邮件 | 注册日期 |
| --- | --- | --- | --- | --- |
| 1 | Alice | Smith | [alice@example.com](mailto:alice@example.com) | 2023-01-15 |
| 2 | Bob | Johnson | [bob.j@example.net](mailto:bob.j@example.net) | 2023-02-20 |
| 3 | Charlie | Davis | [charlie.d@mail.org](mailto:charlie.d@mail.org) | 2023-01-10 |
| 4 | Diana | Brown | [diana@sample.com](mailto:diana@sample.com) | 2023-03-05 |

- `customer_id` 是此表的主键 (PK)。

**2. 订单表 (`Orders`)**

此表记录客户订单。

| 订单ID (主键) | 客户ID (外键) | 产品ID (外键) | 订单日期 | 数量 |
| --- | --- | --- | --- | --- |
| 101 | 1 | 501 | 2023-03-10 | 1 |
| 102 | 3 | 502 | 2023-03-12 | 2 |
| 103 | 1 | 503 | 2023-03-15 | 1 |
| 104 | 2 | 501 | 2023-03-16 | 3 |
| 105 | 4 | 504 | 2023-03-18 | 1 |
| 106 | 3 | 501 | 2023-03-20 | 1 |

- `order_id` 是主键 (PK)。
- `customer_id` 是一个外键 (FK)，引用 `Customers(customer_id)`。
- `product_id` 是一个外键 (FK)，引用 `Products(product_id)`。

**3. 产品表 (`Products`)**

此表列出可用的产品。

| 产品ID (主键) | 产品名称 | 类别 | 价格 |
| --- | --- | --- | --- |
| 501 | 笔记本电脑 | 科技 | 1200 |
| 502 | 键盘 | 科技 | 75 |
| 503 | 咖啡杯 | 家居 | 15 |
| 504 | 台灯 | 家居 | 45 |
| 505 | 无线鼠标 | 科技 | 25 |

- `product_id` 是主键 (PK)。

以下是这些表如何关联的视觉表示：

> 关系图显示 `Customers` 和 `Products` 表分别通过外键 `customer_id` 和 `product_id` 链接到 `Orders` 表。

现在，让我们来编写一些查询！

### 练习 1：获取客户姓名和订单日期

**目标：** 获取一个列表，显示每位客户的名以及他们所下每个订单的日期。

**方法：** 我们需要来自 `Customers`（名）和 `Orders`（订单日期）的数据。这些表通过 `customer_id` 关联。我们将在该公共列上使用 `INNER JOIN`。

**查询：**

```sql
SELECT
    Customers.first_name,
    Orders.order_date
FROM
    Customers
INNER JOIN
    Orders ON Customers.customer_id = Orders.customer_id;
```

**说明：**

- `SELECT Customers.first_name, Orders.order_date`: 指定我们想要从每张表中获取的列。我们使用 `TableName.ColumnName` 语法进行明确指定，特别是当列名可能在两张表中都存在时（尽管此处 `first_name` 和 `order_date` 的情况并非如此）。
- `FROM Customers`: 指定第一张表。
- `INNER JOIN Orders`: 指定我们想要连接的第二张表。
- `ON Customers.customer_id = Orders.customer_id`: 这是连接条件。它告诉数据库匹配 `Customers` 表中 `customer_id` 与 `Orders` 表中 `customer_id` 相同的行。只有匹配的行才会包含在结果中。

**预期输出：**

| 名 | 订单日期 |
| --- | --- |
| Alice | 2023-03-10 |
| Charlie | 2023-03-12 |
| Alice | 2023-03-15 |
| Bob | 2023-03-16 |
| Diana | 2023-03-18 |
| Charlie | 2023-03-20 |

### 练习 2：使用表别名

**目标：** 获取 `order_id`、客户的 `email` 和 `order_date`。使用表别名以求简洁。在输出中将 `email` 列重命名为 `customer_email`。

**方法：** 我们再次连接 `Customers` 和 `Orders`。这次，我们将使用简短的别名（`c` 代表 `Customers`，`o` 代表 `Orders`），使查询更易于阅读和编写。我们还为 `email` 使用列别名。

**查询：**

```sql
SELECT
    o.order_id,
    c.email AS customer_email,
    o.order_date
FROM
    Customers AS c
INNER JOIN
    Orders AS o ON c.customer_id = o.customer_id;
```

**说明：**

- `Customers AS c`: 将别名 `c` 分配给 `Customers` 表。
- `Orders AS o`: 将别名 `o` 分配给 `Orders` 表。
- `o.order_id`、`c.email`、`o.order_date`: 我们使用别名（`c.` 和 `o.`）来引用列，而不是完整的表名。
- `c.email AS customer_email`: 使用列别名将结果集中的 `email` 列重命名为 `customer_email`。
- `ON c.customer_id = o.customer_id`: 连接条件现在使用表别名。

**预期输出：**

| 订单ID | 客户电子邮件 | 订单日期 |
| --- | --- | --- |
| 101 | [alice@example.com](mailto:alice@example.com) | 2023-03-10 |
| 102 | [charlie.d@mail.org](mailto:charlie.d@mail.org) | 2023-03-12 |
| 103 | [alice@example.com](mailto:alice@example.com) | 2023-03-15 |
| 104 | [bob.j@example.net](mailto:bob.j@example.net) | 2023-03-16 |
| 105 | [diana@sample.com](mailto:diana@sample.com) | 2023-03-18 |
| 106 | [charlie.d@mail.org](mailto:charlie.d@mail.org) | 2023-03-20 |

### 练习 3：连接三张表

**目标：** 获取一个组合列表，显示客户的名、所订购产品的名称和订购数量。

**方法：** 我们需要来自所有三张表的信息：`Customers`（名）、`Products`（产品名称）和 `Orders`（数量）。我们需要两个 `INNER JOIN` 子句：一个将 `Customers` 连接到 `Orders`（使用 `customer_id`），另一个将 `Orders` 连接到 `Products`（使用 `product_id`）。我们将再次使用表别名。

**查询：**

```sql
SELECT
    c.first_name,
    p.product_name,
    o.quantity
FROM
    Customers AS c
INNER JOIN
    Orders AS o ON c.customer_id = o.customer_id
INNER JOIN
    Products AS p ON o.product_id = p.product_id;
```

**说明：**

- 我们首先通过 `customer_id` 连接 `Customers` (`c`) 和 `Orders` (`o`)。
- 然后，我们添加另一个 `INNER JOIN` 以引入 `Products` 表 (`p`)。
- 第二个 `ON` 子句（`o.product_id = p.product_id`）根据 `product_id` 将第一次连接的结果（包含 `o.product_id`）连接到 `Products` 表。
- `SELECT` 列表从所有三个使用了别名的表（`c`、`p`、`o`）中提取列。

**预期输出：**

| 名 | 产品名称 | 数量 |
| --- | --- | --- |
| Alice | 笔记本电脑 | 1 |
| Charlie | 键盘 | 2 |
| Alice | 咖啡杯 | 1 |
| Bob | 笔记本电脑 | 3 |
| Diana | 台灯 | 1 |
| Charlie | 笔记本电脑 | 1 |

### 练习 4：组合 JOIN、WHERE 和 ORDER BY

**目标：** 找出客户ID为1（Alice）所下所有订单的 `order_id` 和 `product_name`。按 `order_id` 对结果排序。

**方法：** 我们需要连接 `Orders` 和 `Products` 以获取与每个订单关联的产品名称。然后，我们使用 `WHERE` 子句筛选结果，仅包含来自 `customer_id = 1` 的订单。最后，我们使用 `ORDER BY` 进行排序。

**查询：**

```sql
SELECT
    o.order_id,
    p.product_name
FROM
    Orders AS o
INNER JOIN
    Products AS p ON o.product_id = p.product_id
WHERE
    o.customer_id = 1
ORDER BY
    o.order_id;
```

**说明：**

- 我们首先通过 `product_id` `INNER JOIN` `Orders` (`o`) 和 `Products` (`p`)。
- `WHERE o.customer_id = 1`: 此子句过滤连接后的结果，仅保留 `Orders` 表中 `customer_id` 为 1 的行。请注意，如果我们也连接了 `Customers` 表，我们也可以使用 `WHERE c.customer_id = 1`，但由于 `customer_id` 已经存在于 `Orders` 表中，因此对于此特定查询目标，连接 `Customers` 并非绝对必要。
- `ORDER BY o.order_id`: 根据 `order_id` 对最终筛选的结果集进行升序排序（这是默认排序方式）。

**预期输出：**

| 订单ID | 产品名称 |
| --- | --- |
| 101 | 笔记本电脑 |
| 103 | 咖啡杯 |

这些实践示例表明 `INNER JOIN` 如何让您将分散在不同表中的关联信息编织在一起。通过熟练掌握连接、使用主键和外键正确指定连接条件以及运用别名，您可以构建有效的查询，从关系数据库中提取有价值的信息。尝试修改这些查询或基于示例数据创建您自己的查询，以便更深入地理解表组合。

## 参考资料

- [JOIN Queries](https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOINS) — PostgreSQL Global Development Group (2023)
  Publisher: PostgreSQL Global Development Group
  涵盖PostgreSQL中SQL JOIN子句（包括INNER JOIN）的基本概念和语法。
- [Learning SQL](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057581/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  本书提供了SQL基础概念的理解，其中有专门的章节解释如何使用各种JOIN操作（包括INNER JOIN）组合来自多个表的数据。
- [FROM (Transact-SQL) - Joining Tables](https://learn.microsoft.com/en-us/sql/t-sql/queries/from-transact-sql?view=sql-server-ver16#joining-tables) — Microsoft (2025)
  Publisher: Microsoft
  详细说明Transact-SQL中FROM语句内JOIN子句的语法和用法，并专门涵盖INNER JOIN的官方文档。

---

[上一节](06-%E4%BD%BF%E7%94%A8%E8%A1%A8%E7%9A%84%E5%88%AB%E5%90%8D.md)
