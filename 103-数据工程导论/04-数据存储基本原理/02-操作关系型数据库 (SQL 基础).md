# 操作关系型数据库 (SQL 基础)

来源：[原文](https://apxml.com/zh/courses/intro-data-engineering/chapter-4-data-storage-fundamentals/working-with-relational-databases)

[返回章节目录](README.md) · [返回课程目录](../README.md)

关系型数据库是数据存储的核心，提供了一种结构化的方式来组织和访问信息。可以将它们想象成组织严密的数字文件柜，数据以表的形式存储。这些表类似于电子表格，行代表单个记录（例如特定客户或产品），列代表这些记录的属性或特征（例如客户姓名或产品价格）。

“关系”部分来源于表之间链接数据的能力。例如，您可能有一个`Customers`表，另一个是`Orders`表。您无需在`Orders`表中重复客户信息，而是可以在它们之间建立关系，通常使用唯一标识符。这种结构化方法确保了数据的一致性并减少了冗余。

用于与关系型数据库通信的标准语言是 **SQL（结构化查询语言）**。几乎每个关系型数据库系统（如 PostgreSQL、MySQL、SQL Server、Oracle）都理解 SQL。作为数据工程师，扎实掌握 SQL 对于提取、转换和加载存储在这些系统中的数据是必不可少的。

### 表、列和行

关系型数据库的核心是**表**。

- **列：** 定义所存储数据的属性。每列都有特定的数据类型（例如文本、数字、日期）。例如，在`Products`表中，列可能包括`product_id`（数字）、`name`（文本）和`price`（数字）。
- **行：** 表示表中的单个记录或条目。每行包含表结构中定义的每个列的值。例如，`Products`表中的一行可能表示一个单独的项，如（'101'，'Laptop'，1200.00）。

```text
Table: 产品
+------------+----------+--------+
| product_id | name     | price  |  <- 列（属性）
+------------+----------+--------+
| 101        | Laptop   | 1200.00|  <- 行（记录 1）
| 102        | Keyboard |   75.00|  <- 行（记录 2）
| 103        | Monitor  |  300.00|  <- 行（记录 3）
+------------+----------+--------+
```

### 主键和外键

为了管理关系并确保唯一性，关系型数据库使用键：

- **主键：** 一个列（或一组列），其值唯一标识表中的每一行。在我们的`Products`表中，`product_id`很可能就是主键。没有两个产品可以拥有相同的`product_id`。
- **外键：** 一个列（或一组列），在一个表中引用另一个表中的主键。这在表之间创建了链接或关系。例如，`Orders`表可能有一个`product_id`列，它指回`Products`表中的`product_id`，从而将每个订单行项目链接到特定产品。

> 示意图说明了`Products`表和`Orders`表之间通过外键建立的关系。

### SQL：交互语言

SQL 允许您在数据库上执行各种操作。这些命令通常分为几类：

1. **数据定义语言 (DDL)：** 用于定义或修改数据库结构。
   - `CREATE TABLE`：创建一个新表。
   - `ALTER TABLE`：修改现有表（例如，添加列）。
   - `DROP TABLE`：删除一个表。
2. **数据操作语言 (DML)：** 用于管理表内的数据。
   - `INSERT`：向表中添加新行（记录）。
   - `UPDATE`：修改现有行。
   - `DELETE`：删除行。
3. **数据查询语言 (DQL)：** 用于检索数据（通常被认为是 DML 的一部分）。
   - `SELECT`：从一个或多个表中获取数据。

### 基本 SQL 操作

让我们使用`Products`表示例来了解一些基本 SQL 命令。

**创建表**
您定义表名及其列以及它们的数据类型。

```sql
CREATE TABLE Products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10, 2)
);
```

此命令创建了包含三列的`Products`表：`product_id`作为整数和主键，`name`作为文本（最多100个字符），以及`price`作为十进制数字。

**插入数据**
向表中添加记录：

```sql
INSERT INTO Products (product_id, name, price)
VALUES (101, 'Laptop', 1200.00);

INSERT INTO Products (product_id, name, price)
VALUES (102, 'Keyboard', 75.00);
```

**查询数据 (`SELECT`)**
这可能是最常用的 SQL 命令。它用于检索数据。

获取所有产品的全部列：

```sql
SELECT * FROM Products;
```

`*`符号是“所有列”的简写。

获取所有产品的名称和价格：

```sql
SELECT name, price FROM Products;
```

**筛选数据 (`WHERE`)**
您可以指定条件以仅检索匹配的行。

查找价格低于 100 美元的产品：

```sql
SELECT name, price
FROM Products
WHERE price < 100.00;
```

这将返回“Keyboard”。

**更新数据 (`UPDATE`)**
修改现有记录。使用`WHERE`子句指定要更新*哪些*行非常重要，否则您可能会更新表中的所有行！

更改“Laptop”的价格：

```sql
UPDATE Products
SET price = 1150.00
WHERE product_id = 101;
```

**删除数据 (`DELETE`)**
从表中删除行。同样，`WHERE`子句对于定位特定行很重要。没有它，您将面临删除所有内容的风险！

删除“Keyboard”产品：

```sql
DELETE FROM Products
WHERE product_id = 102;
```

### 关系型数据库的优点

关系型数据库受欢迎的原因有几个：

- **数据完整性：** 诸如主键、外键和约束之类的功能有助于确保数据准确性和一致性。
- **ACID 合规性：** 它们通常遵循 ACID 属性（原子性、一致性、隔离性、持久性），这些属性确保数据库事务可靠地处理。这对于处理财务或关键操作数据的系统来说非常重要。
  - *原子性*：确保事务中的所有操作要么全部成功完成，要么全部不完成。
  - *一致性*：保证事务将数据库从一个有效状态带到另一个有效状态。
  - *隔离性*：确保并发事务互不干扰。
  - *持久性*：确保一旦事务提交，即使在系统故障的情况下也保持提交状态。
- **结构化数据处理：** 它们擅长管理定义明确的结构化数据。
- **成熟技术：** SQL 和关系型数据库已经存在数十年，从而带来了稳定技术、大量文档和庞大社区。

### 常见使用场景

您会发现关系型数据库为以下系统提供支持：

- 在线事务处理 (OLTP) 系统，如电子商务网站、银行应用程序和预订系统。
- 商业智能 (BI) 和报告应用程序，其中分析结构化数据。
- 许多需要持久化、结构化存储的 Web 应用程序。

### 局限性

尽管功能强大，关系型数据库也有局限性：

- **可伸缩性：** 传统关系型数据库的横向扩展（增加更多机器）与某些 NoSQL 替代方案相比可能更复杂，尤其是在极高负载下。
- **灵活性：** 它们不太适合处理非结构化数据（如文本文档、图像）或结构频繁变化的数据。预先定义模式是标准做法。

理解如何使用 SQL 与关系型数据库进行交互是数据工程师的一项基本技能。它们通常是数据提取到管道的源系统，或是清理和转换后的数据加载以供分析的目标系统。在后面的实践部分，您将亲自动手创建一张简单的表。

## 参考资料

- [Learning SQL](https://www.oreilly.com/library/view/learning-sql-3rd-edition/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本学习不同关系数据库系统 SQL 语法和操作的实用指南。
- [PostgreSQL Documentation: The SQL Language](https://www.postgresql.org/docs/current/tutorial-sql.html) — PostgreSQL Global Development Group (2024)
  官方文档，提供了 PostgreSQL 中 SQL 语言的概述和教程。

---

[上一节](01-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E6%95%B0%E6%8D%AE%E5%AD%98%E5%82%A8.md) · [下一节](03-NoSQL%20%E6%95%B0%E6%8D%AE%E5%BA%93%E7%AE%80%E4%BB%8B.md)
