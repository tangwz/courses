---
course: "introduction-to-databases"
chapter: "querying-data-sql"
lesson: "delete-statement-removing-data"
sourceId: 1600
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/delete-statement-removing-data"
title: "DELETE 语句：删除数据"
description: "学习如何使用 `DELETE` 语句从表中删除行，通常与 `WHERE` 子句一起使用。"
order: 7
plots: []
sourceHash: "059d07016bd60bc0ee3bc1b22c83eec06153c20c440ecf59b14e0e9c151c87a9"
sourceCorrections: []
---

SQL 提供了多种数据管理的基本操作，包括获取数据、添加新数据和修改现有数据。另一个主要的数据操作是删除数据。为此，SQL 提供了 `DELETE` 语句。

`DELETE` 语句用于从表中删除一行或多行。与 `UPDATE` 修改现有数据不同，`DELETE` 会完全移除整个记录。

### 基本语法

`DELETE` 语句最简单的形式如下所示：

```sql
DELETE FROM table_name;
```

**警告：** 如上所示，在没有 `WHERE` 子句的情况下执行 `DELETE` 将从指定 `table_name` 中删除**所有行**。此操作通常不可逆，因此，如果使用，应极其谨慎。大多数数据库系统都需要特定权限才能执行如此影响很大的操作，并且通常你只想删除特定行。

### 使用 WHERE 删除特定行

若要只删除符合特定条件的某些行，你必须包含一个 `WHERE` 子句。这是使用 `DELETE` 语句最常见且最安全的方式。`WHERE` 子句的工作方式与 `SELECT` 和 `UPDATE` 语句完全相同，允许你指定用于确定要删除行的条件。

语法如下：

```sql
DELETE FROM table_name
WHERE condition;
```

在这里，`table_name` 是你想要从中删除行的表，`condition` 指定了应删除哪些行。只有满足 `condition` 为真的行才会被删除。

### 示例

我们使用一个示例 `Products` 表来展示 `DELETE` 的工作方式。

**Products 表：**

| ProductID | ProductName | Category | Price | StockQuantity |
| --- | --- | --- | --- | --- |
| 101 | Eco Mug | Kitchenware | 12.50 | 55 |
| 102 | Wireless Mouse | Electronics | 25.00 | 30 |
| 103 | Notebook | Stationery | 3.75 | 150 |
| 104 | USB C Cable | Electronics | 8.00 | 75 |
| 105 | Desk Lamp | Home Goods | 35.00 | 20 |
| 106 | Stapler | Stationery | 5.50 | 0 |

**示例 1：按 ID 删除单行**

假设“台灯”（ProductID 105）已停产。你可以使用其唯一的 `ProductID` 删除它：

```sql
DELETE FROM Products
WHERE ProductID = 105;
```

执行此语句后，`ProductID` 为 105 的行将从 `Products` 表中永久删除。

**示例 2：根据条件删除多行**

假设你想要删除“文具”类别中所有库存为零（`StockQuantity` 为 0）的产品。

```sql
DELETE FROM Products
WHERE Category = 'Stationery' AND StockQuantity = 0;
```

此命令将找到 `Category` 为“文具”*且* `StockQuantity` 为 0 的行。在我们的示例数据中，这将删除“订书机”（ProductID 106）。

### 谨慎操作

`DELETE` 语句功能强大且具有永久性。一旦行被删除，没有备份的情况下，恢复它们可能很困难或不可能。

**一个推荐的安全步骤：**

在执行带有 `WHERE` 子句的 `DELETE` 语句之前，通常一个好的做法是首先运行一个使用*完全相同* `WHERE` 子句的 `SELECT` 语句。这使你能够预览哪些行将受到 `DELETE` 命令的影响。

例如，在运行删除缺货文具的命令之前：

```sql
-- 首先，预览要删除的行
SELECT *
FROM Products
WHERE Category = 'Stationery' AND StockQuantity = 0;

-- 如果结果符合预期，则执行 DELETE
-- DELETE FROM Products
-- WHERE Category = 'Stationery' AND StockQuantity = 0;
```

这个预览步骤有助于避免意外删除错误的数据。在执行 `DELETE` 命令之前，务必仔细检查你的 `WHERE` 子句，尤其是在生产环境中。

随着 `DELETE` 的学习，你现在拥有了基本工具（`SELECT`、`INSERT`、`UPDATE`、`DELETE`），可以使用 SQL 在关系数据库中执行最常见的数据操作任务。下一节将提供练习这些命令的机会。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Abraham Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education
  一本标准的学术教科书，提供了数据库基础知识的广泛介绍，包括SQL数据操作语言（DML）中的DELETE等操作。
- [Sams Teach Yourself SQL in 24 Hours](https://www.informit.com/store/sams-teach-yourself-sql-in-24-hours-9780137543069) — Ryan Stephens (2021)
  Publisher: Sams Publishing; Pages: 624 Pages
  一本实用的、适合初学者的SQL学习指南，其中包含对DELETE语句用法清晰的解释和丰富的示例。
