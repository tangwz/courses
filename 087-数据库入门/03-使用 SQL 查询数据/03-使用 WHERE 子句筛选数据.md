# 使用 WHERE 子句筛选数据

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/filtering-data-where-clause)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 `SELECT` 语句可以从表中获取数据，但您通常不需要*每一行*数据。设想一个有数千条记录的 `Customers` 表；您可能只想查看特定城市的客户，或在某个日期之后注册的客户。获取所有数据并手动筛选既低效又不切实际。

这就是 `WHERE` 子句发挥作用的地方。它作为 `SELECT` 语句的过滤器，让您能够指定行必须满足的条件，以便包含在结果集中。可以把它看作是为您数据获取设置的标准。

### 基本语法

`WHERE` 子句在 `SELECT` 语句中位于 `FROM` 子句之后：

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

`condition` 部分是您定义筛选逻辑的地方。该条件对表中的每一行进行真假判断。只有条件为真的行才会被返回。

### 使用比较运算符

定义条件最常见的方法是使用比较运算符。这些运算符将列中的值与指定值进行比较。假设我们有一个简单的 `Products` 表：

| 产品ID | 名称 | 类别 | 价格 | 库存量 |
| --- | --- | --- | --- | --- |
| 1 | Laptop | Electronics | 1200 | 50 |
| 2 | Coffee Mug | Home Goods | 15 | 200 |
| 3 | T-Shirt | Clothing | 25 | 150 |
| 4 | Smartphone | Electronics | 800 | 75 |
| 5 | Notebook | Stationery | 5 | 500 |
| 6 | Jeans | Clothing | 60 | 100 |

以下是标准比较运算符以及使用我们 `Products` 表的示例：

- **`=` (等于):** 查找列值与指定值完全匹配的行。请注意，文本值（字符串）通常用单引号 (`' '`) 括起来。

  ```sql
  -- 查找 'Clothing' 类别的产品
  SELECT Name, Price
  FROM Products
  WHERE Category = 'Clothing';
  ```

  *结果:* T恤, 牛仔裤
- **`<>` 或 `!=` (不等于):** 查找列值与指定值*不*匹配的行。这两个运算符通常都有效，但 `<>` 是标准 SQL。

  ```sql
  -- 查找不在 'Electronics' 类别的产品
  SELECT Name, Category
  FROM Products
  WHERE Category <> 'Electronics';
  ```

  *结果:* 咖啡杯, T恤, 笔记本, 牛仔裤
- **`>` (大于):** 查找列值在数值上或字母顺序上大于指定值的行。

  ```sql
  -- 查找价格高于100美元的产品
  SELECT Name, Price
  FROM Products
  WHERE Price > 100;
  ```

  *结果:* 笔记本电脑, 智能手机
- **`<` (小于):** 查找列值小于指定值的行。

  ```sql
  -- 查找库存量小于100件的产品
  SELECT Name, StockQuantity
  FROM Products
  WHERE StockQuantity < 100;
  ```

  *结果:* 笔记本电脑, 智能手机
- **`>=` (大于或等于):** 查找列值大于或等于指定值的行。

  ```sql
  -- 查找价格为800美元或更高的产品
  SELECT Name, Price
  FROM Products
  WHERE Price >= 800;
  ```

  *结果:* 笔记本电脑, 智能手机
- **`<=` (小于或等于):** 查找列值小于或等于指定值的行。

  ```sql
  -- 查找价格为25美元或更低的产品
  SELECT Name, Price
  FROM Products
  WHERE Price <= 25;
  ```

  *结果:* 咖啡杯, T恤, 笔记本

### 使用逻辑运算符组合条件

通常，您需要基于多个条件进行筛选。SQL 提供了逻辑运算符（`AND`、`OR`、`NOT`）来在 `WHERE` 子句中组合多个条件。

- **`AND`:** 要求所有关联条件都为真，行才能被包含。

  ```sql
  -- 查找价格低于1000美元的“电子产品”类别产品
  SELECT Name, Price
  FROM Products
  WHERE Category = 'Electronics' AND Price < 1000;
  ```

  *结果:* 智能手机（它是“电子产品”*并且*价格低于1000美元）
- **`OR`:** 要求至少一个关联条件为真，行才能被包含。

  ```sql
  -- 查找属于“服装”类别或价格低于10美元的产品
  SELECT Name, Category, Price
  FROM Products
  WHERE Category = 'Clothing' OR Price < 10;
  ```

  *结果:* T恤, 笔记本, 牛仔裤（T恤和牛仔裤符合类别条件，笔记本符合价格条件）
- **`NOT`:** 反转条件的结果。它常与其他运算符（如 `IN` 或 `LIKE`，接下来会讨论）一起使用，但也可否定简单的比较。

  ```sql
  -- 查找不是“服装”类别的产品
  SELECT Name, Category
  FROM Products
  WHERE NOT Category = 'Clothing';
  -- 这等同于：WHERE Category <> 'Clothing'
  ```

  *结果:* 笔记本电脑, 咖啡杯, 智能手机, 笔记本

混合使用 `AND` 和 `OR` 时，您可以使用括号 `()` 来控制求值顺序，就像在数学中一样。括号内的条件会首先被求值。

```sql
-- 查找电子产品 或 (价格高于50美元的服装类产品)
SELECT Name, Category, Price
FROM Products
WHERE Category = 'Electronics' OR (Category = 'Clothing' AND Price > 50);
```

*结果:* 笔记本电脑, 智能手机, 牛仔裤

### 其他有用的筛选运算符

SQL 提供更多专门的运算符来处理常见的筛选任务：

- **`BETWEEN`:** 检查值是否在指定范围内（包含边界值）。

  ```sql
  -- 查找价格在50美元到100美元之间（包含）的产品
  SELECT Name, Price
  FROM Products
  WHERE Price BETWEEN 50 AND 100;
  ```

  *结果:* 牛仔裤
- **`IN`:** 检查值是否与提供列表中的任何值匹配。这通常比使用多个 `OR` 条件更简洁。

  ```sql
  -- 查找“家居用品”、“文具”或“服装”类别的产品
  SELECT Name, Category
  FROM Products
  WHERE Category IN ('Home Goods', 'Stationery', 'Clothing');
  ```

  *结果:* 咖啡杯, T恤, 笔记本, 牛仔裤
- **`LIKE`:** 对字符串值进行简单的模式匹配。它使用通配符：

  - `%`: 代表零个、一个或多个字符。
  - `_`: 代表恰好一个字符。

  ```sql
  -- 查找名称以 'L' 开头的产品
  SELECT Name
  FROM Products
  WHERE Name LIKE 'L%';
  ```

  *结果:* 笔记本电脑

  ```sql
  -- 查找名称以 't' 结尾的产品
  SELECT Name
  FROM Products
  WHERE Name LIKE '%t';
  ```

  *结果:* 笔记本电脑, T恤, 笔记本

  ```sql
  -- 查找名称第二个字母是 'o' 的产品
  SELECT Name
  FROM Products
  WHERE Name LIKE '_o%';
  ```

  *结果:* 咖啡杯, 笔记本

### 数据类型很重要

请记住，在编写条件时，您需要遵循列的数据类型：

- **字符串 (文本):** 用单引号括起值（例如，`'Electronics'`）。大小写敏感性（例如，`'Electronics'` 是否与 `'electronics'` 匹配）取决于具体的数据库系统及其配置。
- **数字:** 不要用引号括起值（例如，`1200`，`15.99`）。
- **日期:** 通常用引号括起，但所需的格式（例如，`'YYYY-MM-DD'`）在不同的数据库系统（如 MySQL、PostgreSQL、SQL Server）之间有很大差异。

有效使用 `WHERE` 子句是处理数据库的根本。它将 `SELECT` 从一个获取所有数据的粗略工具，转变为一个基于特定标准精确提取所需信息的工具。随着您的练习，您会发现它对于提出有意义的数据问题是不可或缺的。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本标准教材，为数据库系统提供坚实的基础，包括对 SQL 查询和数据操作的详细介绍。
- [PostgreSQL 16 Documentation: SELECT Command](https://www.postgresql.org/docs/16/sql-select.html) — The PostgreSQL Global Development Group (2023)
  PostgreSQL 的官方文档，提供了 SQL SELECT 语句（包括 WHERE 子句及其运算符）的精确语法和行为解释。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本针对 SQL 新手的实用指南，提供清晰的解释和大量示例，说明如何有效地查询、过滤和操作数据。

---

[上一节](02-%60SELECT%60%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E8%8E%B7%E5%8F%96%E6%95%B0%E6%8D%AE.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20ORDER%20BY%20%E5%AF%B9%E7%BB%93%E6%9E%9C%E6%8E%92%E5%BA%8F.md)
