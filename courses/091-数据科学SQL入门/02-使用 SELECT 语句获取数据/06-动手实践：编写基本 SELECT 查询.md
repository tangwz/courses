# 动手实践：编写基本 SELECT 查询

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-2-retrieving-data-select/practice-basic-select)

[返回章节目录](README.md) · [返回课程目录](../README.md)

编写基本的 `SELECT` 查询包括指定列、使用别名和限制结果，需要进行动手实践。通过实际例子进行操作有助于巩固理解，并建立编写 SQL 查询的信心。

在这些练习中，我们假设您正在使用一个包含几个简单表的数据库：`Customers`、`Products` 和 `Orders`。让我们简单看一下它们的结构：

- **`Customers` 表：** 包含客户信息。
  - 列：`CustomerID`、`FirstName`、`LastName`、`Email`、`City`、`Country`
- **`Products` 表：** 包含产品信息。
  - 列：`ProductID`、`ProductName`、`Category`、`Price`
- **`Orders` 表：** 包含订单信息。
  - 列：`OrderID`、`CustomerID`、`OrderDate`、`TotalAmount`

现在我们开始编写一些查询。

### 练习 1：选择特定列

您的第一个任务是获取 `Customers` 表中所有客户的名字、姓氏和电子邮件地址。

**查询：**

```sql
SELECT
  FirstName,
  LastName,
  Email
FROM
  Customers;
```

**解释：**

- `SELECT FirstName, LastName, Email`：这部分精确指定了您要获取哪些列。我们列出所需的列名，并用逗号分隔。
- `FROM Customers`：这表示您要从哪个表中获取数据。

**预期结果（示例）：**

| FirstName | LastName | Email |
| --- | --- | --- |
| Anna | Smith | [anna.s@example.com](mailto:anna.s@example.com) |
| Ben | Jones | [ben.jones@sample.org](mailto:ben.jones@sample.org) |
| Clara | Davis | [c.davis@mail.net](mailto:c.davis@mail.net) |
| ... | ... | ... |

此查询只返回 `Customers` 表中存在的每一行的三个指定列。

### 练习 2：选择所有列并限制数量

接下来，假设您需要快速查看最初几条产品记录。获取 `Products` 表中列出的前 5 个产品的所有可用信息。

**查询：**

```sql
SELECT
  *
FROM
  Products
LIMIT 5;
```

**解释：**

- `SELECT *`：星号（`*`）是一种从表中选择所有列的简写方式，无需单独列出它们。
- `FROM Products`：指定我们正在查询 `Products` 表。
- `LIMIT 5`：此子句将输出限制为数据库遇到的前 5 行。除非使用 `ORDER BY` 子句（稍后介绍），否则确切顺序可能不同，但 `LIMIT` 保证返回的最大行数。请注意，在某些 SQL 方言中，用于限制结果的具体关键字可能略有不同（例如，SQL Server 中的 `TOP 5`，Oracle 中的 `FETCH FIRST 5 ROWS ONLY`）。我们将使用 `LIMIT`，因为它在 PostgreSQL 和 MySQL 等系统中很常用。

**预期结果（示例）：**

| ProductID | ProductName | Category | Price |
| --- | --- | --- | --- |
| 1 | Laptop | Electronics | 1200.00 |
| 2 | Keyboard | Accessories | 75.00 |
| 3 | Mouse | Accessories | 25.50 |
| 4 | Monitor | Electronics | 300.00 |
| 5 | Web Development Book | Books | 49.99 |

此查询获取所有列（`ProductID`、`ProductName`、`Category`、`Price`），但仅限于找到的前 5 条产品行。

### 练习 3：使用列别名以求清晰

有时，原始列名不适合用于报告或分析。让我们从 `Products` 表中获取产品名称及其价格，但将列分别显示为 `Item` 和 `Cost`。

**查询：**

```sql
SELECT
  ProductName AS Item,
  Price AS Cost
FROM
  Products;
```

**解释：**

- `SELECT ProductName AS Item`：我们选择 `ProductName` 列，但使用 `AS` 关键字在输出中为其分配别名 `Item`。
- `Price AS Cost`：类似地，选择 `Price` 列并为其指定别名 `Cost`。
- `FROM Products`：我们仍在查询 `Products` 表。

**预期结果（示例）：**

| Item | Cost |
| --- | --- |
| Laptop | 1200.00 |
| Keyboard | 75.00 |
| Mouse | 25.50 |
| Monitor | 300.00 |
| Web Development Book | 49.99 |
| Data Science Handbook | 59.95 |
| ... | ... |

结果集现在使用别名 `Item` 和 `Cost` 作为列标题，这取决于上下文 (context)，可能使输出更具用户友好性。

### 练习 4：结合使用别名和限制结果

让我们结合所学内容。假设您只需要少量订单标识符及其日期列表。从 `Orders` 表中获取任意 3 份订单的 `OrderID` 和 `OrderDate`。将 `OrderID` 显示为 `OrderNumber`，将 `OrderDate` 显示为 `DatePlaced`。

**查询：**

```sql
SELECT
  OrderID AS OrderNumber,
  OrderDate AS DatePlaced
FROM
  Orders
LIMIT 3;
```

**解释：**

此查询演示了如何结合本章中的多个内容：

- `SELECT OrderID AS OrderNumber, OrderDate AS DatePlaced`：选择特定列并使用 `AS` 分配别名（`OrderNumber`、`DatePlaced`），以提高可读性。
- `FROM Orders`：将 `Orders` 表指定为数据源。
- `LIMIT 3`：将输出限制为数据库为此查询返回的前 3 行。

**预期结果（示例）：**

| OrderNumber | DatePlaced |
| --- | --- |
| 101 | 2023-10-26 |
| 102 | 2023-10-27 |
| 103 | 2023-10-27 |

此实用查询有效地获取了一小部分标记 (token)清晰的订单数据。

这些练习涵盖了使用 `SELECT` 获取数据的核心方法，包括选择特定或所有列、使用别名重命名列以及限制结果的大小。随着您的学习，您将在这些基础上执行更复杂的过滤、排序和计算。尝试这些查询，尝试更改列名、别名和 `LIMIT` 值，看看结果如何变化。

## 参考资料

- [SELECT](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2024)
  详细解释了PostgreSQL中SELECT语句的完整语法和用法，包括选择特定列、使用别名以及LIMIT子句。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057563/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本全面的指南，涵盖了基本的SQL查询语法，包括如何选择数据、使用列别名以及限制返回的行数。

---

[上一节](05-%E9%99%90%E5%88%B6%E6%9F%A5%E8%AF%A2%E7%BB%93%E6%9E%9C.md) · [下一节](../03-%E6%95%B0%E6%8D%AE%E7%AD%9B%E9%80%89%E4%B8%8E%E6%8E%92%E5%BA%8F/01-WHERE%20%E5%AD%90%E5%8F%A5%E7%AD%9B%E9%80%89%E6%95%B0%E6%8D%AE%E7%AE%80%E4%BB%8B.md)
