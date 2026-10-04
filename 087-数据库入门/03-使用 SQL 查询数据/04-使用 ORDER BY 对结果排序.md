# 使用 ORDER BY 对结果排序

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/sorting-results-order-by)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当你使用 `SELECT` 语句获取数据时，数据库会以其认为高效的顺序返回行，这通常显得随机。它可能是数据插入的顺序，也可能取决于数据的内部存储方式。然而，你经常需要以特定顺序展示这些数据，例如按名称字母顺序、按价格数字顺序或按日期时间顺序。

这就是 `ORDER BY` 子句派上用场的地方。它允许你指定一个或多个列，结果集应根据这些列进行排序。你需要将 `ORDER BY` 子句添加到 `SELECT` 语句的末尾。

### 基本排序：升序

最简单的排序形式是升序（A到Z，0到9）。这是 `ORDER BY` 的默认行为。

我们假设有一个 `Products` 表如下：

| ProductID | ProductName | Price |
| --- | --- | --- |
| 1 | Laptop | 1200 |
| 2 | Mouse | 25 |
| 3 | Keyboard | 75 |
| 4 | Monitor | 300 |
| 5 | Webcam | 50 |

如果你运行一个简单的 `SELECT * FROM Products;` 查询，顺序是不确定的。要按 `ProductName` 字母顺序对这些产品排序，你可以使用：

```sql
SELECT ProductID, ProductName, Price
FROM Products
ORDER BY ProductName;
```

这个查询告诉数据库获取所有产品，然后根据 `ProductName` 列的值排列结果，再返回给你。因为升序是默认值，你不需要明确指定它。输出结果如下：

| ProductID | ProductName | Price |
| --- | --- | --- |
| 3 | Keyboard | 75 |
| 1 | Laptop | 1200 |
| 4 | Monitor | 300 |
| 2 | Mouse | 25 |
| 5 | Webcam | 50 |

你可以使用 `ASC` 关键字明确指定升序，尽管它是可选的，因为它是默认值：

```sql
SELECT ProductID, ProductName, Price
FROM Products
ORDER BY ProductName ASC; -- ASC 在此处是可选的
```

这会产生与前一个查询完全相同的结果。

### 降序排序

如果你想看到产品从最贵到最便宜的列表怎么办？为此，你需要按降序排序（Z到A，9到0）。你可以通过在 `ORDER BY` 子句中的列名后添加 `DESC` 关键字来实现。

```sql
SELECT ProductID, ProductName, Price
FROM Products
ORDER BY Price DESC;
```

这个查询根据 `Price` 列对结果进行排序，将它们从最高价排到最低价：

| ProductID | ProductName | Price |
| --- | --- | --- |
| 1 | Laptop | 1200 |
| 4 | Monitor | 300 |
| 3 | Keyboard | 75 |
| 5 | Webcam | 50 |
| 2 | Mouse | 25 |

### 按多列排序

有时，仅按一列排序是不够的。考虑一个 `Employees` 表：

| EmployeeID | FirstName | LastName | Department | Salary |
| --- | --- | --- | --- | --- |
| 101 | Alice | Smith | Sales | 60000 |
| 102 | Bob | Johnson | Engineering | 75000 |
| 103 | Charlie | Smith | Engineering | 80000 |
| 104 | David | Williams | Sales | 55000 |
| 105 | Eve | Brown | Engineering | 70000 |

如果你只按 `Department` 排序，同一部门的员工将以未定义的顺序出现。为了创建一个更有序的列表，你可能希望首先按 `Department` 字母顺序排序，然后在每个部门内部，按 `LastName` 字母顺序排序。

你可以通过在 `ORDER BY` 子句中列出多个列（用逗号分隔）来实现这一点。数据库首先按列出的第一列排序，然后，对于第一列值相同的行，它会按第二列排序，以此类推。

```sql
SELECT EmployeeID, FirstName, LastName, Department
FROM Employees
ORDER BY Department ASC, LastName ASC;
```

你可以省略 `ASC`，因为它是默认值，但为了清晰起见，我们在此包含它。结果如下：

| EmployeeID | FirstName | LastName | Department |
| --- | --- | --- | --- |
| 105 | Eve | Brown | Engineering |
| 102 | Bob | Johnson | Engineering |
| 103 | Charlie | Smith | Engineering |
| 101 | Alice | Smith | Sales |
| 104 | David | Williams | Sales |

注意看，所有“Engineering”部门的员工都排在“Sales”部门的员工前面。在“Engineering”内部，“Brown”排在“Johnson”前面，而“Johnson”排在“Smith”前面。在“Sales”内部，“Smith”排在“Williams”前面。

你也可以混合使用不同列的升序和降序。例如，要按部门字母顺序排序，但在每个部门内部按薪水从高到低排列员工：

```sql
SELECT EmployeeID, FirstName, LastName, Department, Salary
FROM Employees
ORDER BY Department ASC, Salary DESC;
```

| EmployeeID | FirstName | LastName | Department | Salary |
| --- | --- | --- | --- | --- |
| 103 | Charlie | Smith | Engineering | 80000 |
| 102 | Bob | Johnson | Engineering | 75000 |
| 105 | Eve | Brown | Engineering | 70000 |
| 101 | Alice | Smith | Sales | 60000 |
| 104 | David | Williams | Sales | 55000 |

### 组合 WHERE 和 ORDER BY

`ORDER BY` 子句在 `WHERE` 子句筛选行之后才应用。它只对符合 `WHERE` 条件的行进行排序。在你的 SQL 语句中，`ORDER BY` 子句必须始终放在 `WHERE` 子句之后。

示例：选择所有价格低于100美元的产品，并按名称排序。

```sql
SELECT ProductID, ProductName, Price
FROM Products
WHERE Price < 100
ORDER BY ProductName ASC;
```

这个查询首先找到所有价格低于100的产品：Mouse (25)、Keyboard (75) 和 Webcam (50)。然后，它按 `ProductName` 对这些筛选后的结果进行排序：

| ProductID | ProductName | Price |
| --- | --- | --- |
| 3 | Keyboard | 75 |
| 2 | Mouse | 25 |
| 5 | Webcam | 50 |

### 数据类型和排序

`ORDER BY` 与常见数据类型配合使用时非常直观：

- **数字：** 按数字顺序排序 (1, 2, 10, 20)。
- **字符串（文本）：** 根据数据库的排序规则（定义字符顺序、大小写敏感性等）按字母顺序排序。“Apple”排在“Banana”前面。“apple”可能排在“Banana”前面或后面，具体取决于大小写敏感性设置。
- **日期/时间：** 按时间顺序排序（'2023-01-01' 排在 '2023-01-02' 前面）。

### 关于性能的说明

虽然极其有用，但对大型结果集进行排序会消耗大量数据库资源（CPU时间以及可能用于临时存储的内存或磁盘空间）。在拥有数百万或数十亿行的表上，没有适当索引的 `ORDER BY` 子句会显著降低查询速度。这是一个更高级的话题，但了解排序并非“免费”是件好事。对于你初次通常会遇到的数据集，其影响通常可以忽略不计。

总之，`ORDER BY` 子句让你控制查询结果的展示顺序。将其与列名以及可选的 `ASC`（默认）或 `DESC` 结合使用，以有意义地排序数据，使其更容易阅读和分析。请记住它位于 `SELECT` 语句中的 `FROM` 和 `WHERE` 子句之后。

## 参考资料

- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057611/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: Ch. 2-3
  掌握SQL基础知识的实用指南，涵盖ORDER BY等子句的语法和用法，以有效进行数据操作和呈现。
- [SELECT](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2023)
  PostgreSQL中SELECT语句的官方文档，提供了在广泛使用的关系型数据库系统中ORDER BY子句的权威且具体的信息。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20WHERE%20%E5%AD%90%E5%8F%A5%E7%AD%9B%E9%80%89%E6%95%B0%E6%8D%AE.md) · [下一节](05-INSERT%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E6%B7%BB%E5%8A%A0%E6%95%B0%E6%8D%AE.md)
