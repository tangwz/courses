# UPDATE 语句：修改数据

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/update-statement-modifying-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据很少保持不变。将数据添加到表中后，通常需要进行修改。例如，顾客可能会更改地址，员工可能获得晋升和加薪，或者订单状态可能需要更新。`UPDATE` 语句正是为此而设计。它允许您更改数据库表中行内的现有数据。

### UPDATE 的基本结构

修改数据的基本语法如下：

```sql
UPDATE table_name
SET column1 = new_value1,
    column2 = new_value2,
    ...
WHERE condition;
```

我们来解释一下其组成部分：

1. **`UPDATE table_name`**: 您首先指定要修改的表。将 `table_name` 替换为表的实际名称（例如，`Employees`、`Products`、`Orders`）。
2. **`SET column1 = new_value1, ...`**: `SET` 子句是您定义更改的地方。您列出要更新的列以及它们应包含的新值。
   - `column1`、`column2` 等是您要更改其值的列的名称。
   - `new_value1`、`new_value2` 等是对应的新值。请确保新值的数据类型与列的数据类型一致（例如，为 `VARCHAR` 列提供文本，为 `INTEGER` 列提供数字）。
   - 您可以通过用逗号分隔每个 `column = value` 对来更新单个列或多个列。
3. **`WHERE condition`**: 这个子句对于精准更新非常重要。它指定表中 *哪些行* 应该被修改。`condition` 的作用方式与您在 `SELECT` 语句中学到的 `WHERE` 子句完全相同。它会筛选行，只有满足条件的行才会被更新。

### 为什么 WHERE 子句如此重要

您可能会想省略 `WHERE` 子句。**对此务必非常小心。** 如果您在没有 `WHERE` 子句的情况下执行 `UPDATE` 语句，数据库会将 `SET` 子句中指定的更改应用到表中的**每一行**。

想象一下不小心将所有员工的工资设置为相同值，或将所有客户的城市更改为相同地点。这种错误可能难以纠正且耗时，尤其是在大型数据库中。

**经验法则：** 除非您完全确定要修改表中的每一行，否则请务必在 `UPDATE` 语句中包含 `WHERE` 子句。通常，一个好的做法是先用相同的 `WHERE` 子句编写 `SELECT` 语句，以验证哪些行 *会* 受到影响，然后再运行实际的 `UPDATE`。

### UPDATE 的使用示例

我们以 `Employees` 表为例，看看 `UPDATE` 的实际应用。

**`Employees` 表：**

| 员工ID | 名 | 姓 | 部门 | 薪水 |
| --- | --- | --- | --- | --- |
| 101 | Alice | Smith | Sales | 60000 |
| 102 | Bob | Jones | IT | 75000 |
| 103 | Charlie | Brown | Sales | 62000 |
| 104 | Diana | Prince | HR | 68000 |

**示例 1：为特定行更新单个列**

假设 Bob Jones（员工ID 102）从 IT 部门调到 Engineering 部门。我们需要更新他的记录。

```sql
UPDATE Employees
SET Department = 'Engineering'
WHERE EmployeeID = 102;
```

- `UPDATE Employees`: 我们指定 `Employees` 表。
- `SET Department = 'Engineering'`: 我们将 `Department` 列设置为新值 'Engineering'。
- `WHERE EmployeeID = 102`: 我们只针对 `EmployeeID` 为 102 的行。

执行此语句后，表会变成这样：

| 员工ID | 名 | 姓 | 部门 | 薪水 |
| --- | --- | --- | --- | --- |
| 101 | Alice | Smith | Sales | 60000 |
| 102 | Bob | Jones | Engineering | 75000 |
| 103 | Charlie | Brown | Sales | 62000 |
| 104 | Diana | Prince | HR | 68000 |

**示例 2：为特定行更新多个列**

现在，假设 Alice Smith（员工ID 101）获得晋升。她的部门变更为 'Sales Management'，薪水增加到 65000。

```sql
UPDATE Employees
SET Department = 'Sales Management',
    Salary = 65000
WHERE EmployeeID = 101;
```

- 我们在 `SET` 子句中列出多个更改，用逗号分隔。
- `WHERE` 子句仍然使用 Alice 唯一的 `EmployeeID` 精准定位她的记录。

The table now becomes:

| 员工ID | 名 | 姓 | 部门 | 薪水 |
| --- | --- | --- | --- | --- |
| 101 | Alice | Smith | Sales Management | 65000 |
| 102 | Bob | Jones | Engineering | 75000 |
| 103 | Charlie | Brown | Sales | 62000 |
| 104 | Diana | Prince | HR | 68000 |

**示例 3：更新多行**

假设公司决定给目前薪水低于 $65000 的所有人加薪$1000。

```sql
UPDATE Employees
SET Salary = Salary + 1000
WHERE Salary < 65000;
```

- 这里，`SET` 子句使用一个表达式：`Salary = Salary + 1000`。这会获取 `Salary` 列中的当前值并加上 1000。
- `WHERE Salary < 65000` 子句识别所有当前薪水满足此条件的员工。在我们当前的表中，这会影响 Charlie Brown（薪水 62000）。

本次更新后：

| 员工ID | 名 | 姓 | 部门 | 薪水 |
| --- | --- | --- | --- | --- |
| 101 | Alice | Smith | Sales Management | 65000 |
| 102 | Bob | Jones | Engineering | 75000 |
| 103 | Charlie | Brown | Sales | 63000 |
| 104 | Diana | Prince | HR | 68000 |

**示例 4：危险区域 - 不带 `WHERE` 的更新（务必极其谨慎！）**

如果您运行以下命令（并且我们强烈建议在未仔细考虑的情况下不要这样做）：

```sql
-- 危险示例 - 除非有明确意图，否则不要运行！
UPDATE Employees
SET Department = 'General';
```

此命令会将表中**每位员工**的 `Department` 更改为 'General'，从而丢失所有先前的部门信息。在执行前务必仔细检查您的 `UPDATE` 语句，特别是 `WHERE` 子句。

`UPDATE` 语句是长期维护数据准确性的强大工具。通过将其与 `SET` 和 `WHERE` 子句谨慎结合，您可以对关系数据库中的现有记录进行精准修改。

## 参考资料

- [Database System Concepts](https://www.mheducation.com/highered/product/database-system-concepts-silberschatz-korth/9780078022159.html) — Abraham Silberschatz, Henry Korth, S. Sudarshan (2020)
  Publisher: McGraw-Hill Education
  一本涵盖数据库系统的基础学术教科书，详细解释了 SQL 数据操作语言（如 UPDATE）的各项操作。
- [UPDATE Statement](https://dev.mysql.com/doc/refman/8.4/en/update.html) — MySQL Documentation (2023)
  Publisher: Oracle
  MySQL UPDATE 语句的官方文档，提供了在 MySQL 环境中其语法规则、示例及应用说明。

---

[上一节](05-INSERT%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E6%B7%BB%E5%8A%A0%E6%95%B0%E6%8D%AE.md) · [下一节](07-DELETE%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E5%88%A0%E9%99%A4%E6%95%B0%E6%8D%AE.md)
