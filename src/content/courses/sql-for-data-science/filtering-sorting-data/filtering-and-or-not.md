---
course: "sql-for-data-science"
chapter: "filtering-sorting-data"
lesson: "filtering-and-or-not"
sourceId: 1579
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/filtering-and-or-not"
title: "使用 AND、OR、NOT 进行筛选"
description: "通过使用 AND、OR 和 NOT 组合多个条件来构建更复杂的筛选逻辑。"
order: 3
plots: []
sourceHash: "a176777c00cc4ae9660c9ad5b8de73a5fcde9c780d5a3b8bf23b9bc6e7e87e4c"
sourceCorrections: []
---

`WHERE` 子句允许指定一个条件来筛选 `SELECT` 语句返回的行。然而，通常筛选需求会更复杂。你可能需要找出同时满足*多个*条件的行，或满足几个条件中*至少一个*的行。逻辑运算符 `AND`、`OR` 和 `NOT` 在这些情况下变得重要。它们允许在一个 `WHERE` 子句中组合多个条件，从而准确定义所需的数据子集。

### 使用 AND 组合条件

当你想获取满足*所有*指定条件的行时，会使用 `AND` 运算符。可以将其视为一个交集：一行必须同时满足条件 1 *和* 条件 2 等，才能包含在结果集中。

基本语法如下所示：

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition1 AND condition2;
```

假设我们有一个 `Products` 表，包含 `product_name`、`category` 和 `price` 等列。如果我们想找出所有属于“电子产品”类别*并且*价格超过 500 美元的产品，我们会使用 `AND`：

```sql
SELECT product_name, category, price
FROM Products
WHERE category = 'Electronics' AND price > 500;
```

只有同时满足 `category = 'Electronics'` 和 `price > 500` *两个*条件的产品才会出现在结果中。如果一个产品属于“电子产品”但价格为 400 美元，它将不被包含。同理，如果一个产品价格为 600 美元但属于“家居用品”类别，它也不会被包含。

如果你需要满足两个以上的条件，可以将多个 `AND` 运算符连接起来：

```sql
SELECT product_name, category, price, stock_quantity
FROM Products
WHERE category = 'Electronics' AND price > 500 AND stock_quantity > 0;
```

此查询找出价格超过 500 美元且目前有库存的电子产品。

### 使用 OR 组合条件

当你希望获取满足指定条件中*至少一个*的行时，会使用 `OR` 运算符。可以将其视为一个并集：一行只需满足条件 1 *或* 条件 2，或任何其他通过 `OR` 连接的条件，即可被包含。

语法与 `AND` 类似：

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition1 OR condition2;
```

再次使用我们的 `Products` 表，假设我们想找出所有属于“电子产品”类别*或*属于“家电”类别的产品。我们使用 `OR`：

```sql
SELECT product_name, category, price
FROM Products
WHERE category = 'Electronics' OR category = 'Appliances';
```

此查询将返回产品，如果它们的类别是“电子产品”，*或*如果它们的类别是“家电”，*或*如果（假设性地）一个产品可以同时属于两者。只要通过 `OR` 连接的条件中*有一个*对给定行是真的，该行就会被包含在结果中。

就像 `AND` 一样，你可以连接多个 `OR` 条件：

```sql
SELECT product_name, category, price
FROM Products
WHERE category = 'Electronics' OR category = 'Appliances' OR price < 50;
```

此查询获取属于“电子产品”类别的产品，*或*属于“家电”类别的产品，*或*任何价格低于 50 美元的产品（无论类别如何）。

### 使用 NOT 否定条件

`NOT` 运算符用于反转条件的逻辑。它选择指定条件为*假*的行。

其语法通常是在你想要否定的条件前放置 `NOT`：

```sql
SELECT column1, column2, ...
FROM table_name
WHERE NOT condition;
```

例如，要找出所有*不*属于“电子产品”类别的产品：

```sql
SELECT product_name, category, price
FROM Products
WHERE NOT category = 'Electronics';
```

或者，对于简单的相等/不相等判断，通常更易读的方式是使用 `<>` 或 `!=` 运算符（表示“不等于”）:

```sql
SELECT product_name, category, price
FROM Products
WHERE category <> 'Electronics'; -- 或者使用 category != 'Electronics'
```

`NOT` 运算符与其他运算符（如 `IN`、`BETWEEN` 或 `LIKE`）结合使用时特别有用。例如，要找出类别*不*是“电子产品”且*不*是“家电”的产品：

```sql
SELECT product_name, category, price
FROM Products
WHERE category NOT IN ('Electronics', 'Appliances');
```

这通常比写 `WHERE NOT (category = 'Electronics' OR category = 'Appliances')` 更清晰。

### 运算符优先级和使用括号

当你在一个 `WHERE` 子句中组合 `AND`、`OR` 和 `NOT` 时，数据库需要知道评估这些条件的顺序。SQL 有一个标准的优先级顺序：

1. 首先评估 `NOT`。
2. 接下来评估 `AND`。
3. 最后评估 `OR`。

如果你不小心，这种优先级顺序有时会导致意外结果。思考这个查询：找出属于“电子产品”类别*且*价格超过 500 美元的产品，*或*仅仅是“新品”的产品。

一个潜在的尝试可能是：

```sql
-- 潜在模糊的查询
SELECT product_name, category, price, status
FROM Products
WHERE category = 'Electronics' AND price > 500 OR status = 'New';
```

因为 `AND` 的优先级高于 `OR`，数据库会将其解释为：
`(category = 'Electronics' AND price > 500) OR status = 'New'`

这意味着它将选择：

- 类别是“电子产品”且价格大于 500 的任何产品。
- *此外*，状态是“新品”的任何产品（无论其类别或价格如何）。

这可能不是你预期的结果。如果你想找出满足 `price > 500` *或* `status = 'New'` 条件中*任何一个*的“电子产品”，你*必须*使用括号 `()` 来覆盖默认优先级并明确你的逻辑：

```sql
-- 使用括号使查询更清晰
SELECT product_name, category, price, status
FROM Products
WHERE category = 'Electronics' AND (price > 500 OR status = 'New');
```

现在，括号内的条件 `(price > 500 OR status = 'New')` 会首先被评估。该查询只会返回同时满足*这两个*条件的产品：

1. 类别*必须*是“电子产品”。
2. 产品还必须满足 `price > 500` *或* `status = 'New'` 中*任何一个*条件。

**最佳实践：** 即使默认优先级与你的意图一致，使用括号也能让你的 `WHERE` 子句更易于阅读和理解，从而减少逻辑错误的可能。当组合使用 `AND` 和 `OR` 时，几乎总是推荐使用括号来明确地分组你的条件。

熟练掌握 `AND`、`OR` 和 `NOT`，以及策略性地使用括号，可以让你构建复杂的筛选逻辑，使你能够从大型数据集中分离出分析所需的精确数据。

## 参考资料

- [SQL SELECT Statement: WHERE Clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-WHERE) — PostgreSQL Global Development Group (2024)
  涵盖 PostgreSQL 中使用 WHERE 子句筛选数据以及使用逻辑运算符组合条件。
- [MySQL 8.0 Reference Manual: Boolean Logic](https://dev.mysql.com/doc/refman/8.0/en/fulltext-boolean.html) — Oracle and/or its affiliates (2024)
  Publisher: Oracle
  阐述 MySQL 中条件表达式的布尔逻辑和运算符（AND、OR、NOT）。
- [Learning SQL: Master SQL Fundamentals](http://www.oreilly.com/catalog/9781492057611) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 377
  SQL 基础知识的综合指南，包含 WHERE 子句和逻辑运算符的说明。
