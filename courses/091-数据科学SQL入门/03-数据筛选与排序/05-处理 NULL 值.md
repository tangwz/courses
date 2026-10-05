# 处理 NULL 值

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/handling-null-values)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据常常包含不完整的信息。有时，某行中特定列的值可能缺失、未知或不适用。SQL 使用一个特殊的标记 (token) `NULL` 来表示这种值的缺失。`NULL` 不同于数字的零 (0)、文本的空字符串 ('') 或任何其他特定值。`NULL` 特指“没有值”。

在使用 `WHERE` 子句筛选数据时，您可能自然地尝试使用标准比较运算符如 `=` 或 `<>` (或 `!=`) 来查找包含或不包含 `NULL` 值的行。例如，您可能会尝试：

```sql
-- 这不会按预期工作
SELECT product_name, price
FROM products
WHERE price = NULL;
```

或者可能：

```sql
-- 这也不会按预期工作
SELECT product_name, price
FROM products
WHERE price <> NULL;
```

令人惊讶的是，这两个查询通常都不会返回您期望的行。这是因为 `NULL` 表示一种未知状态。将任何东西与一个未知值进行比较，甚至是另一个未知值，结果都是 `UNKNOWN`。在 `WHERE` 子句中，计算结果为 `UNKNOWN` 的条件被视为 `FALSE`，这意味着这些行*不*会包含在结果集中。

那么，如何正确筛选值缺失或存在的行呢？SQL 为此提供了特定的运算符：`IS NULL` 和 `IS NOT NULL`。

### 使用 IS NULL 查找缺失值

要选择特定列包含 `NULL` 值的行，您可以使用 `IS NULL` 运算符。

**语法：**

```sql
SELECT column1, column2, ...
FROM table_name
WHERE column_name IS NULL;
```

**例子：**

假设我们有一个 `customers` 表，并且有些客户没有提供他们的电子邮件地址。要找到这些客户，您会这样写：

```sql
SELECT customer_id, first_name, last_name
FROM customers
WHERE email IS NULL;
```

这个查询将返回 `customers` 表中所有 `email` 列明确包含 `NULL` 标记 (token)的行，这表明电子邮件地址缺失。

### 使用 IS NOT NULL 查找存在的值

相反，要选择特定列*确实*包含值（即，它*不*是 `NULL`）的行，您可以使用 `IS NOT NULL` 运算符。

**语法：**

```sql
SELECT column1, column2, ...
FROM table_name
WHERE column_name IS NOT NULL;
```

**例子：**

如果您只想获取*已*提供电子邮件地址的客户，您会使用：

```sql
SELECT customer_id, first_name, last_name, email
FROM customers
WHERE email IS NOT NULL;
```

这个查询只返回 `customers` 表中 `email` 列有值存在的行，不包括那些标记 (token)为 `NULL` 的行。

### 结合 NULL 检查与其他条件

您可以使用 `AND` 和 `OR` 轻松地将 `IS NULL` 和 `IS NOT NULL` 与其他筛选条件结合起来。

**例子：**

让我们找出所有缺货（数量为 0）或价格未设置（`price` 为 `NULL`）的产品：

```sql
SELECT product_name, price, quantity
FROM products
WHERE quantity = 0 OR price IS NULL;
```

**例子：**

找出所有来自“加利福尼亚”并且*已*提供电子邮件地址的客户：

```sql
SELECT first_name, last_name, email
FROM customers
WHERE state = 'CA' AND email IS NOT NULL;
```

"知道并正确使用 `IS NULL` 和 `IS NOT NULL` 在处理通常含有缺失信息的数据集时非常重要。能够根据数据的存在与否来识别或排除行，是数据清洗和准备分析的必要步骤。对于数据科学家来说，妥善识别和处理缺失值是确保分析结果可信度的重要组成部分。"

## 参考资料

- [PostgreSQL 16.2 Documentation - 4.2. Value Expressions - NULL](https://www.postgresql.org/docs/current/sql-expressions.html#SQL-EXPRESSIONS-NULL-LOGIC) — The PostgreSQL Global Development Group (2024)
  提供 NULL 行为、三值逻辑以及在广泛使用的 SQL 数据库系统中比较规则的官方说明。
- [Learning SQL: Generate, Manipulate, and Retrieve Data](https://books.google.com/books/about/Learning_SQL_Generate_Manipulate_and_Ret.html?id=YwK2DwAAQBAJ) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本广泛使用的 SQL 学习书籍，清晰地解释了 NULL 值等核心概念、它们在比较中的行为以及使用 IS NULL 和 IS NOT NULL 的正确处理方法。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本经典的学术教材，提供了数据库系统的理论基础，包括 NULL 的形式定义、三值逻辑以及它们在关系数据库理论和查询处理中的作用。

---

[上一节](04-%E4%BD%BF%E7%94%A8%20IN%20%E5%92%8C%20BETWEEN%20%E8%BF%9B%E8%A1%8C%E7%AD%9B%E9%80%89.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20LIKE%20%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%BC%8F%E5%8C%B9%E9%85%8D.md)
