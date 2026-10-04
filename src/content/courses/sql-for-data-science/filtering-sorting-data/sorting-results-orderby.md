---
course: "sql-for-data-science"
chapter: "filtering-sorting-data"
lesson: "sorting-results-orderby"
sourceId: 1591
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/sorting-results-orderby"
title: "使用 ORDER BY 排序结果"
description: "学习如何根据一个或多个列，按升序 (ASC) 或降序 (DESC) 排序查询结果。"
order: 7
plots: []
sourceHash: "93d2d72df0733f6260c79ea9b9da71fd4b140430bc0bd3f62cb7a7c227bc0bcd"
sourceCorrections: []
---

SQL 查询可以用来选择特定列并使用 `WHERE` 子句过滤行。然而，查询返回的行并没有确定的顺序，除非明确指定。通常，你会希望看到排序后的结果，也许是为了找出最高或最低值，或者仅仅是为了更清晰地展示信息。`ORDER BY` 子句提供了这个功能。它允许你指定用于排序结果集的列。

### 基本排序

`ORDER BY` 子句通常添加到 `SELECT` 语句的末尾，通常在 `FROM` 和 `WHERE` 子句（如果存在）之后。你指定要排序的列。

假设我们有一个 `products` 表，其中包含 `product_name`、`category` 和 `price` 等列。如果你想按产品名称的字母顺序排列产品列表，你可以这样写：

```sql
SELECT product_name, category, price
FROM products
ORDER BY product_name;
```

默认情况下，`ORDER BY` 按升序排序数据（文本从 A 到 Z，数字从小到大）。

### 指定排序方向：ASC 和 DESC

尽管升序 (`ASC`) 是默认的，但你可以明确指定它，也可以要求降序 (`DESC`)。

- `ASC`：升序（从低到高，A 到 Z）。如果你不指定任何内容，这是默认值。
- `DESC`：降序（从高到低，Z 到 A）。

要将产品从最贵到最便宜排序，你可以使用 `DESC`：

```sql
SELECT product_name, category, price
FROM products
ORDER BY price DESC;
```

如果你想明确按产品名称的字母顺序（升序）排序，你可以这样写：

```sql
SELECT product_name, category, price
FROM products
ORDER BY product_name ASC;
```

尽管 `ASC` 在这里是可选的，但明确指定它可以提高可读性。

### 按多个列排序

如果你按一个列排序，而该列有多个行具有相同的值，会发生什么？例如，如果你按 `category` 排序产品，*相同*类别中的产品如何排序？默认情况下，它们的顺序不确定。

你可以指定次要（或第三个等）排序列来控制组内的顺序。在 `ORDER BY` 子句中列出这些列，用逗号分隔。排序是按顺序进行的：数据首先按列出的第一个列排序。然后，对于第一个列值相同的任何行，数据将按列出的第二个列排序，依此类推。

让我们首先按类别（字母顺序）排序产品，然后，在每个类别内，按价格（从高到低）排序：

```sql
SELECT product_name, category, price
FROM products
ORDER BY category ASC, price DESC;
```

在此查询中：

1. 所有行首先按 `category` 升序排序（例如，'Electronics' 在 'Home Goods' 之前）。
2. 在每个类别中（例如，所有 'Electronics' 产品），行然后按 `price` 降序排序（最贵的电子产品在该类别组中首先出现）。

你可以在同一个 `ORDER BY` 子句中为不同的列混合使用 `ASC` 和 `DESC`。

### 将 ORDER BY 与 WHERE 结合使用

`ORDER BY` 子句与 `WHERE` 子句结合使用效果很好。数据库首先根据 `WHERE` 条件过滤行，然后根据 `ORDER BY` 的规定对*结果*行进行排序。

例如，要找出 'Electronics' 类别中价格低于 500 美元的所有产品，并按价格从低到高排序：

```sql
SELECT product_name, category, price
FROM products
WHERE category = 'Electronics' AND price < 500
ORDER BY price ASC; -- ASC is optional here
```

请记住 `SELECT` 语句中子句的标准顺序：

1. `SELECT` 列...
2. `FROM` 表...
3. `WHERE` 条件...
4. `ORDER BY` 列...
5. `LIMIT` 数量...（如果使用，通常在最后或 `ORDER BY` 之后）

### 关于 NULL 值的说明

`NULL` 值在排序时的处理方式在不同的数据库系统（如 PostgreSQL、MySQL、SQL Server）之间可能有所不同。有些系统在升序排序时将 `NULL` 值放在前面，降序排序时放在后面，而另一些系统则可能相反，或者提供 `NULLS FIRST` 或 `NULLS LAST` 等语法来明确控制。对于基本排序，只需注意 `NULL` 值可能出现在排序结果的开头或结尾。

排序是使查询结果易于理解的基本组成部分。`ORDER BY` 子句为你提供了对展示顺序的精确控制，允许你根据一个或多个列以升序或降序逻辑地安排数据。这对于查找极端值、直观地对相关项进行分组或准备报告都非常有价值。

## 参考资料

- [SELECT Statement](https://www.postgresql.org/docs/current/sql-select.html) — PostgreSQL Global Development Group (2025)
  Publisher: PostgreSQL Global Development Group
  官方文档，详细介绍了 `SELECT` 语句及其 `ORDER BY` 子句，包括排序方向、多列排序以及 `NULL` 值的具体处理方式。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本关于数据库系统的标准学术教科书，在关系型数据库的背景下涵盖了SQL基础知识，包括 `ORDER BY` 子句。
- [Intro to SQL: Querying and managing data](https://www.khanacademy.org/computing/computer-programming/sql) — Khan Academy (2023)
  Publisher: Khan Academy
  一个入门级的在线课程，清晰地解释了SQL基础概念，包括如何使用 `ORDER BY` 子句排序查询结果，并提供了实用示例。
