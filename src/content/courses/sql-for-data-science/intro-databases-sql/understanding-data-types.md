---
course: "sql-for-data-science"
chapter: "intro-databases-sql"
lesson: "understanding-data-types"
sourceId: 1552
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-1-intro-databases-sql/understanding-data-types"
title: "理解数据类型"
description: "介绍 SQL 数据库中使用的基本数据类型，如 INTEGER、VARCHAR、DATE 等。"
order: 3
plots: []
sourceHash: "8f0beb1a40cb55d0bd119032d6a8d139f8d8e73d019af4558d240d367e32298b"
sourceCorrections: []
---

让我们巩固对表、列和行的理解。在定义表中的列时，我们不仅会给它命名；还会明确它能存放什么*种类*的数据。这种设定被称为列的**数据类型**。

可以把数据类型看作是你列的规则。就像你不会把汤装在筛子里一样，你也不会把“Hello World”这样的文本存储在一个只用于数字的列中。定义数据类型非常重要，因为它能确保：

1. **数据完整性：** 避免意外插入错误类型的数据（例如，将文本插入日期列）。
2. **存储效率：** 数据库系统能根据值的类型，了解要为每个值分配多少空间。
3. **正确操作：** 它使数据库能够执行有意义的操作（例如，你可以对数字求和，但不能对文本求和）。

尽管不同数据库系统（如PostgreSQL、MySQL、SQL Server、SQLite）之间的具体名称和功能可能略有不同，但核心思想和常见类型大体一致。以下是你经常会遇到的主要类别：

### 数值类型

这些类型用于处理数值数据。

- **INTEGER（或 INT）：** 用于存储整数（不带小数点的数字）。示例包括用户ID（如 `101`、`102`）、数量（`5`、`100`）或计数。它们可以是正数或负数。还存在 `SMALLINT`、`TINYINT`、`BIGINT` 等变体，用于存储较小或较大范围的整数，以优化存储空间。
- **DECIMAL（或 NUMERIC）：** 用于需要精确度的数字，特别是货币或金融计算。通常需要指定总位数（精度）和小数点后的位数（标度）。例如，`DECIMAL(10, 2)` 可以存储最多10位数字的数字，其中2位用于小数部分（例如，`12345678.99`）。
- **FLOAT、REAL、DOUBLE PRECISION：** 这些用于近似值数字，通常用于科学计算，在这些计算中，精确度不如处理非常宽泛的值范围（包括非常小或非常大的数字）那么关键。它们使用浮点表示法，有时可能导致小的舍入差异。

### 字符串（字符）类型

这些类型用于存储文本数据。

- **VARCHAR(n)：** 表示*可变字符*。这是最常见的文本类型之一。它存储长度可变的字符串，最长可达由 `n` 指定的长度。例如，`VARCHAR(255)` 可以存储0到255个字符长的任何文本字符串。数据库只使用实际存储字符所需的空间，加上少量额外开销。当文本长度变化很大时使用此类型，例如姓名、地址或描述。
- **CHAR(n)：** 表示*固定字符*。此类型存储固定长度的字符串，长度由 `n` 指定。如果你插入的字符串短于 `n`，数据库通常会用空格填充它以达到准确的长度 `n`。如果字符串更长，它可能会被截断或引发错误，这取决于数据库系统。在某些情况下，如果数据长度始终相同（例如，使用 `CHAR(2)` 存储两位国家代码，如 `US`、`CA`、`GB`），`CHAR` 的查询速度可能会稍快。
- **TEXT：** 用于存储长篇文本数据，如文章、评论或详细描述，其最大长度可能超过 `VARCHAR` 通常能轻松支持的范围。`TEXT` 的确切长度限制在不同数据库系统之间差异很大，但通常非常大。

### 日期和时间类型

这些类型专门用于存储时间信息。

- **DATE：** 只存储日期（年、月和日）。示例：`2023-10-27`。
- **TIME：** 只存储一天中的时间（时、分和秒，有时包含小数秒）。示例：`14:35:10`。
- **TIMESTAMP（或 DATETIME）：** 同时存储日期和时间信息。示例：`2023-10-27 14:35:10.123`。这些类型常用于记录事件，例如记录订单下达时间或记录最后更新时间。某些系统存储相对于UTC（协调世界时）的时间戳信息，这有助于管理时区。

### 布尔类型

- **BOOLEAN：** 表示逻辑上的 `TRUE` 或 `FALSE` 值。某些数据库系统没有特定的 `BOOLEAN` 类型，可能会转而使用小的整数类型（如 `TINYINT(1)`，其中 `0` 表示 `FALSE`，`1` 表示 `TRUE`）。用于标志或状态指示器，例如 `is_active`、`has_paid` 或 `email_verified`。

### 表结构中的数据类型可视化

让我们看看这些类型在一个简单的 `Customers` 表中如何应用。

> 一张图表显示了 `Customers` 表中的列及其对应的 SQL 数据类型。`customer_id` 通常是 `INTEGER` 类型，用作唯一标识符（主键）。姓名和邮箱使用 `VARCHAR` 存储可变长度文本。`signup_date` 使用 `DATE`。`total_spent` 使用 `DECIMAL` 存储精确的货币值，而 `is_premium` 使用 `BOOLEAN` 存储真/假状态。

在设计表时选择适当的数据类型是重要一步。它能确保你的数据存储正确且高效，使后续使用SQL查询和分析变得更轻松、更可靠。随着课程的进行，你将看到这些类型如何影响我们编写查询的方式，尤其是在筛选和执行计算时。

## 参考资料

- [PostgreSQL: Documentation - Chapter 8. Data Types](https://www.postgresql.org/docs/current/datatype.html) — The PostgreSQL Global Development Group (2023)
  提供PostgreSQL中所有数据类型的全面详细信息，包括其用法和存储特性。这有助于用户理解通用数据类型原则的具体实现。
- [MySQL 8.0 Reference Manual :: 11 Data Types](https://dev.mysql.com/doc/refman/8.0/en/data-types.html) — Oracle and/or its affiliates (2023)
  Publisher: Oracle
  MySQL支持的数据类型的权威指南，对于理解不同数据库系统中类型定义的异同至关重要。
- [Learning SQL](https://books.google.com/books/about/Learning_SQL.html?id=9w9cDwAAQBAJ) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media, Inc.; Pages: 380
  一本广受认可的SQL基础入门教材，包含关于数据类型及其在数据库设计中实际应用的章节。第3版。
