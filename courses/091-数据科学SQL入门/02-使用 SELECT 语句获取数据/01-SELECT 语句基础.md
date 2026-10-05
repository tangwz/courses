# SELECT 语句基础

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-2-retrieving-data-select/basic-select-statement)

[返回章节目录](README.md) · [返回课程目录](../README.md)

关系数据库将信息整齐地组织在表中，就像带有行和列的电子表格一样。为了将这些信息取出用于分析、报告或任何其他目的，结构化查询语言 (SQL) 提供了 `SELECT` 语句。这可以说是 SQL 中最基本且最常用的命令。可以把它当作您向数据库提问的主要工具。

`SELECT` 语句最基本的形式需要两个主要部分：

1. **`SELECT`**: 后跟您要获取数据的列列表。
2. **`FROM`**: 后跟这些列所在的表名。

结构如下所示：

```sql
SELECT column_name1, column_name2
FROM table_name;
```

我们来分析一下：

- `SELECT` 和 `FROM` 是 SQL 关键字。它们告诉数据库引擎您打算执行哪种操作。虽然 SQL 通常不区分大小写（这意味着 `SELECT`、`select` 和 `SeLeCt` 通常被视为相同），但将 SQL 关键字写成大写是常见做法，也有助于提高可读性。
- `column_name1, column_name2` 代表您感兴趣的特定列。如果您想要多个列，请用逗号将它们分开。
- `table_name` 是包含您希望查询的数据的表的标识符。
- 末尾的分号 (`;`) 标志着 SQL 语句的结束。虽然并非所有数据库系统或接口都严格要求使用它，但使用它是标准做法，有助于避免歧义，尤其是在一起编写多个语句时。

假设我们有一个名为 `Products` 的简单表，其中存储了库存物品的信息：

| 产品ID | 产品名称 | 类别 | 价格 |
| --- | --- | --- | --- |
| 101 | 小部件 | 小工具 | 19.99 |
| 102 | 小玩意 | 小工具 | 25.50 |
| 103 | 链轮 | 零件 | 5.75 |
| 104 | 玩意儿 | 小工具 | 12.00 |

如果您只想获取所有产品名称的列表，可以编写以下查询：

```sql
SELECT product_name
FROM Products;
```

对数据库执行此查询将返回一个结果集，其中只包含 `product_name` 列的值：

| 产品名称 |
| --- |
| 小部件 |
| 小玩意 |
| 链轮 |
| 玩意儿 |

如果您需要每种产品的名称和价格，可以在 `SELECT` 关键字后列出这两个列，用逗号分隔：

```sql
SELECT product_name, price
FROM Products;
```

此查询将生成一个包含两列的结果集：

| 产品名称 | 价格 |
| --- | --- |
| 小部件 | 19.99 |
| 小玩意 | 25.50 |
| 链轮 | 5.75 |
| 玩意儿 | 12.00 |

这种基本的 `SELECT ... FROM ...` 结构是 SQL 中几乎所有数据获取的构成部分。在接下来的章节中，我们将在此基础上学习如何选择所有列、重命名输出中的列、限制结果数量，并最终以更复杂的方式筛选、排序和组合数据。现在，请着重理解这个核心语法：指定您想要 *什么* (`SELECT column(s)`) 和它 *在哪里* (`FROM table`)。

## 参考资料

- [SELECT](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2025)
  官方文档，详细解释了SELECT语句，包括其基本形式和各种子句。
- [Learning SQL: Master the Fundamentals of SQL and Harness the Power of Databases](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057611/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 377
  一本广受好评的书籍，提供了SQL的全面介绍，从SELECT语句等核心概念开始。
- [SELECT Statement](https://dev.mysql.com/doc/refman/8.4/en/select.html) — Oracle and/or its affiliates (2024)
  Publisher: Oracle
  MySQL官方关于SELECT语句的文档，涵盖其语法和基本用法。

---

[上一节](../01-%E6%95%B0%E6%8D%AE%E5%BA%93%E5%92%8C%20SQL%20%E5%85%A5%E9%97%A8/05-%E6%90%AD%E5%BB%BA%E4%BD%A0%E7%9A%84%E7%8E%AF%E5%A2%83%EF%BC%88%E5%8F%AF%E9%80%89%EF%BC%89.md) · [下一节](02-%E9%80%89%E6%8B%A9%E7%89%B9%E5%AE%9A%E5%88%97.md)
