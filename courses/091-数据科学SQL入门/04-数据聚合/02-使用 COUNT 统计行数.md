# 使用 COUNT 统计行数

来源：[原文](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/counting-rows-count)

[返回章节目录](README.md) · [返回课程目录](../README.md)

原始数据通常需要汇总，以便提供有价值的信息。最基本的汇总之一就是简单地计数：我们有多少客户？昨天下了多少订单？售出了多少种不同的产品？SQL 为此提供了一个专门的聚合函数：`COUNT`。

`COUNT` 函数统计满足特定条件的行数。我们来看看它的常见形式。

### 统计所有行：`COUNT(*)`

`COUNT` 最直接的用法是使用星号 (`*`)。`COUNT(*)` 仅统计一个表或由 `WHERE` 子句定义的查询结果中的总行数。

**语法：**

```sql
SELECT COUNT(*)
FROM table_name;
```

**示例：**

假设我们有一个 `Customers` 表，如下所示：

| 客户ID | 名 | 姓 | 电子邮件 | 城市 |
| --- | --- | --- | --- | --- |
| 1 | Alice | Smith | [alice.s@example.com](mailto:alice.s@example.com) | London |
| 2 | Bob | Jones | [bob.j@example.com](mailto:bob.j@example.com) | Manchester |
| 3 | Charlie | Brown | NULL | London |
| 4 | Diana | Prince | [diana.p@example.com](mailto:diana.p@example.com) | Birmingham |
| 5 | Edward | King | [edward.k@example.com](mailto:edward.k@example.com) | London |

要查找表中的客户总数，您将运行：

```sql
SELECT COUNT(*)
FROM Customers;
```

**结果：**

| 统计(\*) |
| --- |
| 5 |

此查询返回 5，因为 `Customers` 表中有五行。

### 统计列中的非 NULL 值：`COUNT(column_name)`

有时，您可能不想统计所有行，而是统计特定列中具有非 NULL 值的行数。为此，您将列名放在括号内。

**语法：**

```sql
SELECT COUNT(column_name)
FROM table_name;
```

**示例：**

使用相同的 `Customers` 表，我们来统计有多少客户提供了电子邮件地址。我们以 `email` 列为目标：

```sql
SELECT COUNT(email)
FROM Customers;
```

**结果：**

| 统计(email) |
| --- |
| 4 |

请注意，结果是 4，而不是 5。这是因为 `COUNT(email)` 会忽略 `email` 列值为 `NULL` 的行（客户 ID 3）。`COUNT(*)` 统计行数，而 `COUNT(column_name)` 统计该列中所有行中的非 NULL 值。

### 统计列中的唯一值：`COUNT(DISTINCT column_name)`

通常，您需要知道一个列中有多少个*唯一*值。例如，我们的客户居住在多少个不同的城市？在 `COUNT` 中使用 `DISTINCT` 关键字可以实现此目的。

**语法：**

```sql
SELECT COUNT(DISTINCT column_name)
FROM table_name;
```

**示例：**

我们来统计我们的 `Customers` 表中不同城市的数量：

```sql
SELECT COUNT(DISTINCT city)
FROM Customers;
```

**结果：**

| 统计(DISTINCT city) |
| --- |
| 3 |

此查询返回 3，因为尽管有 5 行，但列出的城市是 'London'、'Manchester'、'London'、'Birmingham' 和 'London'。不同的城市是 London、Manchester 和 Birmingham。如果列中存在 `NULL` 值，则不计入 `DISTINCT` 统计。

### 在 COUNT 中使用别名

`COUNT` 结果的默认列名（例如 `COUNT(*)` 或 `COUNT(email)`）并不总是具有描述性。您可以使用 `AS` 关键字分配一个更具意义的名称，这会创建一个别名。

**示例：**

我们来重命名我们之前的示例的输出：

```sql
-- 客户总数
SELECT COUNT(*) AS total_customers
FROM Customers;

-- 有电子邮件的客户数量
SELECT COUNT(email) AS customers_with_email
FROM Customers;

-- 不同的客户城市数量
SELECT COUNT(DISTINCT city) AS distinct_customer_cities
FROM Customers;
```

**结果：**

| total\_customers |
| --- |
| 5 |

| customers\_with\_email |
| --- |
| 4 |

| distinct\_customer\_cities |
| --- |
| 3 |

使用别名可以使输出更清晰、更容易理解，特别是当您开始在单个查询中组合多个聚合函数时。



[交互图表：客户数据统计](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/counting-rows-count#plot-r1tyq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "客户数据统计"
    },
    "xaxis": {
      "title": {
        "text": "统计类型"
      }
    },
    "yaxis": {
      "title": {
        "text": "记录数量"
      }
    },
    "barmode": "group"
  },
  "data": [
    {
      "type": "bar",
      "name": "客户总数",
      "x": [
        "统计"
      ],
      "y": [
        5
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "bar",
      "name": "有电子邮件的客户",
      "x": [
        "统计"
      ],
      "y": [
        4
      ],
      "marker": {
        "color": "#12b886"
      }
    },
    {
      "type": "bar",
      "name": "不同城市",
      "x": [
        "统计"
      ],
      "y": [
        3
      ],
      "marker": {
        "color": "#fab005"
      }
    }
  ]
}
```

</details>



> 来自 `Customers` 表的不同统计比较：总行数 vs. 非 NULL 电子邮件 vs. 不同城市。

### 将 `COUNT` 与 `WHERE` 结合使用

像 `COUNT` 这样的聚合函数是在 `WHERE` 子句过滤行*之后*应用的。这意味着您可以统计满足特定条件的行数。

**示例：**

有多少客户居住在 London？

```sql
SELECT COUNT(*) AS london_customers
FROM Customers
WHERE city = 'London';
```

**结果：**

| london\_customers |
| --- |
| 3 |

数据库首先过滤 `Customers` 表以仅包含 `city` 为 'London' 的行，然后 `COUNT(*)` 统计那些过滤后的行。

`COUNT` 函数是汇总数据的基本构成要素。您会经常使用它，通常与 `GROUP BY` 子句结合使用（我们很快就会介绍），以统计不同类别中的项目，例如统计每个客户下的订单数量。

## 参考资料

- [PostgreSQL: Documentation: 16: 9.20. Aggregate Functions](https://www.postgresql.org/docs/current/functions-aggregate.html) — PostgreSQL Global Development Group (2023)
  Publisher: PostgreSQL Global Development Group
  解释了COUNT聚合函数及其不同形式（COUNT(*)、COUNT(表达式)、COUNT(DISTINCT 表达式)），以及它与NULL值的行为，与本节内容直接相关。
- [COUNT (Transact-SQL)](https://learn.microsoft.com/en-us/sql/t-sql/functions/count-transact-sql) — Microsoft Learn (2024)
  提供了SQL Server中COUNT函数的具体语法和示例，涵盖了本节讨论的所有形式，包括与DISTINCT和WHERE子句的用法。

---

[上一节](01-%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20SUM%20%E8%AE%A1%E7%AE%97%E6%80%BB%E5%92%8C.md)
