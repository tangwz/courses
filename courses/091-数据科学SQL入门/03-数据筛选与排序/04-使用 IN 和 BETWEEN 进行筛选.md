# 使用 IN 和 BETWEEN 进行筛选

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/filtering-in-between)

[返回章节目录](README.md) · [返回课程目录](../README.md)

SQL 提供了专用运算符来增强数据筛选功能。当条件需要检查一个值是否属于某个特定选项集合或是否在一个连续范围内时，可以使用 `IN` 和 `BETWEEN` 运算符。这些运算符提供了一种更简洁易读的方式来执行此类筛选任务。

### 使用 IN 检查值列表

假设你想获取特定列与几个可能值之一匹配的行数据。例如，你可能想找出所有居住在“加利福尼亚”、“内华达”或“亚利桑那”的客户。你可以使用多个 `OR` 条件来编写：

```sql
SELECT customer_name, state
FROM customers
WHERE state = 'California' OR state = 'Nevada' OR state = 'Arizona';
```

这种方法可行，但如果你的值列表变长，查询语句可能会变得冗长且难以阅读。`IN` 运算符提供了一种更简洁的方式来表达相同的逻辑。它允许你指定一个值列表进行检查。

基本语法是：

`WHERE column_name IN (value1, value2, value3, ...)`

使用 `IN`，之前的查询会变得更加紧凑：

```sql
SELECT customer_name, state
FROM customers
WHERE state IN ('California', 'Nevada', 'Arizona');
```

这个查询与使用 `OR` 的查询结果完全相同，但通常更易于编写和理解，尤其是在列表较长时。`IN` 运算符适用于多种数据类型，包括数字和字符串。

这是一个选择具有特定状态码订单的例子：

```sql
SELECT order_id, order_date, status_code
FROM orders
WHERE status_code IN (100, 205, 410);
```

你还可以将 `IN` 与 `AND` 或 `OR` 的其他条件结合使用。例如，要找出居住在特定州 *并且* 在某个日期之后注册的客户：

```sql
SELECT customer_name, state, signup_date
FROM customers
WHERE state IN ('California', 'Nevada', 'Arizona')
  AND signup_date > '2023-01-01';
```

就像你可以将 `NOT` 与其他运算符一起使用一样，你可以使用 `NOT IN` 来查找列值与列表中任何值都 *不* 匹配的行：

```sql
SELECT customer_name, state
FROM customers
WHERE state NOT IN ('California', 'Nevada', 'Arizona');
```

这将返回除加利福尼亚、内华达和亚利桑那之外所有州的客户。

### 使用 BETWEEN 检查范围

另一种常见的筛选情况是选择列值落在特定范围（包括边界）内的行。例如，查找价格在 $50 到$100 之间（含两端）的产品。你可以使用 `AND` 和比较运算符：

```sql
SELECT product_name, price
FROM products
WHERE price >= 50 AND price <= 100;
```

同样，这种方法也完全可行。然而，SQL 提供了 `BETWEEN` 运算符，作为表达范围的更直接方式。

语法是：

`WHERE column_name BETWEEN lower_bound AND upper_bound`

一个重要细节是 `BETWEEN` 是**包含边界的**：它包含列值等于 `lower_bound` 或 `upper_bound` 的行。

使用 `BETWEEN`，产品价格查询如下：

```sql
SELECT product_name, price
FROM products
WHERE price BETWEEN 50 AND 100;
```

这通常被认为是表达范围更易读的方式。`BETWEEN` 常用于数字和日期。

日期示例（语法可能因具体 SQL 数据库而略有不同）：

```sql
SELECT event_name, event_date
FROM events
WHERE event_date BETWEEN '2024-06-01' AND '2024-06-30';
```

此查询选择发生在 2024 年 6 月的事件，包括 6 月 1 日和 6 月 30 日的事件。

类似于 `IN`，你还可以用 `NOT BETWEEN` 来排除一个范围：

```sql
SELECT product_name, price
FROM products
WHERE price NOT BETWEEN 50 AND 100;
```

这会返回价格低于 $50 或高于$100 的产品。

`IN` 和 `BETWEEN` 都简化了常见的筛选任务。与使用多个 `OR` 条件或带有 `AND` 的复合不等式相比，它们使你的 `WHERE` 子句更短，并且通常更易于理解。熟练使用它们是编写高效 SQL 查询的重要一步。

## 参考资料

- [MySQL 8.0 Reference Manual :: 12.5.3.2 Comparison Operators](https://dev.mysql.com/doc/refman/8.0/en/comparison-functions-and-operators.html) — Oracle and/or its affiliates (2020)
  Publisher: Oracle
  MySQL比较运算符的官方文档，详细介绍了IN和BETWEEN运算符及其使用方法。
- [Learning SQL, Third Edition](https://www.oreilly.com/library/view/learning-sql-third-edition/9781492057611/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 51-70
  一本广泛使用的SQL入门书籍，涵盖了基本的过滤技术，包括IN和BETWEEN运算符。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20AND%E3%80%81OR%E3%80%81NOT%20%E8%BF%9B%E8%A1%8C%E7%AD%9B%E9%80%89.md) · [下一节](05-%E5%A4%84%E7%90%86%20NULL%20%E5%80%BC.md)
