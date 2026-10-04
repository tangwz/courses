# INSERT 语句：添加数据

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/insert-statement-adding-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

SQL 命令，如 `SELECT`，允许用户从数据库中检索信息，而 `WHERE` 和 `ORDER BY` 则可以进一步精细化这些查询。虽然这些命令对于数据检索是必不可少的，但问题仍然存在：数据最初是如何进入表中的呢？`INSERT` 语句提供了答案。它是向表中添加新行或记录的基本 SQL 命令。

可以把表想象成一个电子表格网格。`SELECT` 从网格中读取行。`INSERT` 则向该网格添加全新的行。

### 基本语法：添加完整行

使用 `INSERT` 最直接的方法是为表中每个列提供一个值，且值要与列的定义顺序完全一致。基本语法如下：

```sql
INSERT INTO table_name
VALUES (value1, value2, value3, ...);
```

让我们分解来看：

1. `INSERT INTO table_name`：这指定了你要添加新行的目标表。将 `table_name` 替换为你的表的实际名称（例如，`Customers`、`Products`）。
2. `VALUES (value1, value2, value3, ...)`：这个子句提供了新行的实际数据。
   - 值必须用括号 `()` 括起来。
   - 值必须按列在表定义中出现的*相同顺序*排列。
   - 你需要为表中的*每个*列提供一个值。
   - 每个值的数据类型必须与对应列的数据类型匹配（例如，为整型列提供数字，为字符型列提供用单引号括起来的文本）。

**示例：**

假设我们有一个 `Products` 表，定义如下：

- `ProductID` (整数，主键)
- `ProductName` (VARCHAR - 即文本)
- `Price` (DECIMAL - 即带小数的数字)
- `StockQuantity` (整数)

要添加一个新产品“Espresso Machine”，其 `ProductID` 为 101，价格为 199.99，库存量为 15，你可以这样写：

```sql
INSERT INTO Products
VALUES (101, 'Espresso Machine', 199.99, 15);
```

执行此命令后，包含此信息的新行将添加到 `Products` 表中。

### 指定列名

依赖精确的列顺序可能有风险。如果表结构后续发生变化（例如，列重新排序或添加新列），你的 `INSERT` 语句可能会失败，甚至更糟，将数据插入错误的列中。

一个更安全、通常也更清晰的方法是明确列出你要填充的列，然后按照相同的顺序列出相应的值。

语法是：

```sql
INSERT INTO table_name (column1, column2, column3, ...)
VALUES (value1, value2, value3, ...);
```

在这里：

1. `(column1, column2, column3, ...)`：你列出要提供数据的具体列。
2. `(value1, value2, value3, ...)`：你提供与所列列对应的值。`value1` 对应 `column1`，`value2` 对应 `column2`，依此类推。

**示例：**

使用相同的 `Products` 表，让我们添加一个价格为 45.50，库存量为 30 的“Coffee Grinder”。我们知道 `ProductID` 是主键，通常数据库可以自动生成这些（我们会简单提及这一点）。目前，我们假设我们*不需要*指定 `ProductID`，或者我们想显式插入它，但为了演示目的可以不按顺序。指定列为我们提供了灵活性。

```sql
INSERT INTO Products (ProductName, Price, StockQuantity, ProductID)
VALUES ('Coffee Grinder', 45.50, 30, 102);
```

或者，如果 `ProductID` 是由数据库自动生成的（一个称为自增长或标识的常见功能）：

```sql
INSERT INTO Products (ProductName, Price, StockQuantity)
VALUES ('Coffee Grinder', 45.50, 30);
```

在第二种情况下，数据库本身会分配下一个可用的 `ProductID`（如果 101 是上一个使用的，则可能是 102）。

使用列列表可以使你的 SQL 语句更具可读性，并且在表结构发生变化时更不容易出错。你只需为你列出的列提供值。任何*未*列出的列将接收其默认值（如果定义了默认值），或者如果没有默认值且不能为空（如 `NOT NULL` 列），则可能导致错误。对于初学者来说，除非你确定列的顺序并且提供了所有值，否则列出所有列是一个好习惯。

### 插入多行

大多数数据库系统允许你使用单个 `INSERT` 语句插入多行，这比为每行运行单独的 `INSERT` 命令更高效。语法通常涉及列出多组值：

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES
  (value1a, value2a, ...),
  (value1b, value2b, ...),
  (value1c, value2c, ...);
```

**示例：**

向我们的 `Products` 表添加另外两个产品（假设 `ProductID` 是自动生成的）：

```sql
INSERT INTO Products (ProductName, Price, StockQuantity)
VALUES
  ('Milk Frother', 25.00, 50),
  ('Pour Over Kettle', 60.75, 22);
```

这个单一的语句向 `Products` 表添加了两条不同的行。

`INSERT` 语句是你填充表的工具。无论你是通过顺序隐式地为所有列提供值，还是明确列出这些列，你都在向数据库添加新的完整记录。请记住匹配数据类型，如果你没有指定列名，还要留意列的顺序。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill; Pages: Chapter 3
  一本综合性教材，涵盖数据库管理系统基本概念，包括SQL数据操作语言（DML）中的INSERT语句。
- [INSERT](https://www.postgresql.org/docs/current/sql-insert.html) — PostgreSQL Global Development Group (2023)
  PostgreSQL `INSERT`语句的官方文档，详细说明其语法、用法及变体。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057631/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: Chapter 4
  学习SQL基础的实用指南，包含对`INSERT`语句清晰的解释和示例。
- [INSERT Statement](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEioVAyVqqpo3tLvftNcSYQ5C2Kf30h0O11HiSauisdhVwZb3_AH7F-etZyIz59Ix5sYG5_oq73bqhkgUVrwxJ1UKiOPrCiPJeet_Acb98JFoUdPdmubRj5B-FG2lM4BCcaSQ==) — Oracle and/or its affiliates (2023)
  Publisher: Oracle
  MySQL `INSERT`语句的官方文档，为MySQL用户提供语法和具体行为。

---

[上一节](04-%E4%BD%BF%E7%94%A8%20ORDER%20BY%20%E5%AF%B9%E7%BB%93%E6%9E%9C%E6%8E%92%E5%BA%8F.md) · [下一节](06-UPDATE%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E4%BF%AE%E6%94%B9%E6%95%B0%E6%8D%AE.md)
