# 使用 LIKE 进行模式匹配

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/pattern-matching-like)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 SQL 中筛选行通常涉及精确匹配 (`=`)、比较 (`>`, `<`)、范围 (`BETWEEN`) 或列表成员 (`IN`)。然而，有时您需要根据*模式*而不是精确值来查找数据。这在处理文本数据时很常见。例如，您可能需要查找来自特定域的所有电子邮件地址、所有以特定前缀开头的产品代码，或所有包含特定字母序列的名称。

SQL 提供了 `LIKE` 操作符来达成这个目的。它用于 `WHERE` 子句中，以在字符串列（如 `VARCHAR`、`TEXT` 等）中搜索指定的模式。

### 使用 `LIKE` 操作符

使用 `LIKE` 的基本语法很简单：

```sql
SELECT column1, column_name
FROM table_name
WHERE column_name LIKE 'pattern';
```

这个“模式”不仅仅是字面字符串；它包含被称为**通配符**的特殊字符，这些字符可以实现灵活的匹配。

### 模式匹配的通配符

与 `LIKE` 操作符一起使用的主要通配符有两种：

1. **百分号 (`%`)**：代表零个、一个或多个字符。它是最灵活的通配符。
2. **下划线 (`_`)**：代表恰好一个字符。

我们通过一些例子来看看它们是如何运作的。假设我们有一个 `products` 表，其中包含 `product_name` 列。

- **查找以 'C' 开头的名称**：
  要查找所有名称以字母 'C' 开头的产品，您可以在 'C' 后面加上 `%`。`%` 匹配 'C' 后面的任何字符序列（或无字符）。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE 'C%';
  ```

  这将匹配 'Chair'、'Computer'、'Cable'、'Cap' 等。
- **查找以 't' 结尾的名称**：
  要查找以 't' 结尾的名称，将 `%` 放在 't' 的前面。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE '%t';
  ```

  这可能匹配 'Shirt'、'Hat'、'Carpet'、'Widget'。
- **查找包含 'Desk' 的名称**：
  要查找名称中任何位置包含 'Desk' 的名称，将 `%` 放在 'Desk' 的前面和后面。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE '%Desk%';
  ```

  这将匹配 'Standing Desk'、'Desk Lamp'、'Office Desk Set'。
- **查找以 'L' 开头并以 'p' 结尾的名称**：
  您可以结合字面字符和通配符。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE 'L%p';
  ```

  这可能匹配 'Laptop'、'Lamp'。

现在我们来看看下划线 (`_`)。

- **查找第二个字母是 'a' 的名称**：
  下划线 `_` 用作恰好一个字符的占位符。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE '_a%';
  ```

  这将匹配 'Table'、'Lamp'、'Keyboard'（因为 'a' 是第二个字母，且 `%` 匹配其余部分）。它*不会*匹配 'Apple'（其中 'a' 是第一个字母）。
- **查找以 'P' 开头的四个字母的名称**：
  您可以使用多个下划线。

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE 'P___';
  ```

  这将匹配 'Pens'、'Pins'，假设这些是仅有的以 'P' 开头的四个字母的产品名称。它不会匹配 'Paper'（5个字母）或 'Pen'（3个字母）。

### 组合通配符

您可以在同一个模式中混合使用 `%` 和 `_` 进行更具体的搜索。

- **查找第二个字母是 'o' 且名称后续包含 'er' 的名称**：

  ```sql
  SELECT product_name
  FROM products
  WHERE product_name LIKE '_o%er%';
  ```

  这可能匹配 'Monitor'、'Power Cord'、'Folder'。

### 大小写敏感性

一个重要之处是，`LIKE` 在大小写敏感性（即 'a' 是否匹配 'A'）方面的行为取决于您使用的具体数据库系统，有时也取决于其配置。

- **PostgreSQL**：`LIKE` 默认区分大小写。它提供 `ILIKE` 用于不区分大小写的匹配。
- **MySQL**：大小写敏感性取决于表的排序规则设置，但通常默认为不区分大小写。
- **SQL Server**：敏感性取决于数据库或列的排序规则。

如果您不确定大小写敏感性，请务必查看您特定数据库系统的文档或进行测试。如果您需要特定的行为（区分大小写或不区分大小写），请在您的数据库系统中查找相应的操作符或函数。

### 查找*不*匹配的内容：`NOT LIKE`

就像其他比较操作符一样，您可以使用 `NOT` 来否定 `LIKE`。要查找所有名称*不*以 'S' 开头的产品，您可以使用：

```sql
SELECT product_name
FROM products
WHERE product_name NOT LIKE 'S%';
```

### 综合应用：示例场景

我们使用一个 `customers` 表，其中包含 `first_name`、`last_name` 和 `email` 列。

- **查找名字以 'J' 开头的客户**：

  ```sql
  SELECT first_name, last_name
  FROM customers
  WHERE first_name LIKE 'J%';
  ```
- **查找电子邮件地址来自 'datastyle.com' 的客户**：

  ```sql
  SELECT first_name, email
  FROM customers
  WHERE email LIKE '%@datastyle.com';
  ```
- **查找姓氏恰好是5个字母长并以 's' 结尾的客户**：

  ```sql
  SELECT last_name
  FROM customers
  WHERE last_name LIKE '____s';
  ```
- **查找名字不包含字母 'a' 的客户（不区分大小写，假设 ILIKE 可用或 LIKE 不区分大小写）**：

  ```sql
  -- 使用 ILIKE（PostgreSQL 示例）
  SELECT first_name
  FROM customers
  WHERE first_name NOT ILIKE '%a%';

  -- 或者如果 LIKE 不区分大小写（MySQL 默认示例）
  SELECT first_name
  FROM customers
  WHERE first_name NOT LIKE '%a%';
  ```

`LIKE` 操作符及其通配符 `%` 和 `_` 结合使用，提供了一种根据模式筛选文本数据的方法，极大地扩展了 `WHERE` 子句的功能。请记住根据您的数据库系统考虑大小写敏感性。

## 参考资料

- [Learning SQL: Master the Fundamentals of Relational Databases](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本全面介绍SQL基础知识的书籍，包括LIKE操作符和使用通配符的多种模式匹配技术。
- [PostgreSQL Documentation: Pattern Matching](https://www.postgresql.org/docs/current/functions-matching.html) — PostgreSQL Global Development Group (2023)
  PostgreSQL模式匹配操作符的官方文档，详细说明了LIKE、用于不区分大小写匹配的ILIKE以及正则表达式。
- [MySQL 8.0 Reference Manual: String Comparison Functions and Operators](https://dev.mysql.com/doc/refman/8.0/en/string-comparison-functions.html) — Oracle and/or its affiliates (2023)
  Publisher: Oracle
  MySQL字符串比较函数的官方文档，详细介绍了LIKE操作符及其在字符集和排序规则方面的影响，这些因素会影响大小写敏感性。

---

[上一节](05-%E5%A4%84%E7%90%86%20NULL%20%E5%80%BC.md) · [下一节](07-%E4%BD%BF%E7%94%A8%20ORDER%20BY%20%E6%8E%92%E5%BA%8F%E7%BB%93%E6%9E%9C.md)
