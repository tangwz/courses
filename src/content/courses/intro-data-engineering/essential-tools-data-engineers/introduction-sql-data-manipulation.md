---
course: "intro-data-engineering"
chapter: "essential-tools-data-engineers"
lesson: "introduction-sql-data-manipulation"
sourceId: 5591
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-6-essential-tools-data-engineers/introduction-sql-data-manipulation"
title: "SQL数据操作入门"
description: "学习SQL如何用于在数据库中查询和操作数据。"
order: 1
plots: []
sourceHash: "43dc3a966d3f38b3ae624955ecd5b69e8325a64cbb21ee0d1b34f06f47f77749"
sourceCorrections: []
---

结构化查询语言，通常称为SQL（常发音为“sequel”或“S-Q-L”），是与关系型数据库交互的标准语言。作为数据工程师，你会发现数据库是管理数据的核心存储位置，而SQL是你与它们沟通的主要工具。可以把它看作是访问和操作表中数据通用遥控器。

你已经在第4章了解了不同的存储系统，包括关系型数据库（如PostgreSQL、MySQL、SQL Server）。这些数据库将数据组织成表，表的结构类似于电子表格，行代表记录，列代表这些记录的属性或特性。SQL提供了一种结构化的方式来向这些数据库提问（查询）并指示它们如何更改所持有的数据（操作）。

### 为何SQL用于数据操作？

数据很少是静态的。作为数据工程师，你将持续需要：

- **获取数据：** 用于检查数据质量、提供数据进行分析或将数据输入其他系统。
- **添加新数据：** 当新信息从不同来源抵达时。
- **修改现有数据：** 用于纠正错误、更新信息或丰富记录。
- **删除数据：** 用于删除重复项、处理过期记录或遵守法规。

SQL提供了专门为这些任务设计的命令。虽然第4章涉及了数据库结构定义（数据定义语言，即DDL，如`CREATE TABLE`），但本节侧重于用于操作这些表*内部*数据的命令（数据操作语言，即DML）。

### SQL数据操作基本命令

SQL中操作数据的主要操作常被称为CRUD（创建、读取、更新、删除）缩写，尽管在SQL术语中，它们主要对应于`INSERT`、`SELECT`、`UPDATE`和`DELETE`。我们逐一来看。

#### 读取数据：`SELECT`语句

`SELECT`语句用于从一个或多个表中获取数据。这可能是你最常使用的SQL命令。

- **选择特定列：** 你可以指定要查看的列。

  ```sql
  SELECT user_id, email, registration_date
  FROM users;
  ```

  此查询仅从名为`users`的表中获取`user_id`、`email`和`registration_date`列。
- **选择所有列：** 星号（`*`）是选择所有列的通配符。

  ```sql
  SELECT *
  FROM users;
  ```

  这会获取`users`表中所有行的所有列。
- **使用`WHERE`筛选数据：** 通常，你只想要符合特定条件的行。`WHERE`子句允许你筛选结果。

  ```sql
  SELECT user_id, email
  FROM users
  WHERE city = 'New York';
  ```

  此查询仅获取`city`列值为'New York'的用户的`user_id`和`email`。

  你可以在`WHERE`子句中使用各种运算符，例如`=`、`>`、`<`、`>=`、`<=`、`!=`（不等于）、`LIKE`（模式匹配），并使用`AND`和`OR`组合条件。

#### 添加数据：`INSERT INTO`语句

当需要向表中添加新数据时，使用`INSERT INTO`语句。

- **语法：** 你指定表、要提供数据的列以及相应的值。

  ```sql
  INSERT INTO users (user_id, email, city, registration_date)
  VALUES (101, 'new.user@example.com', 'London', '2023-10-26');
  ```

  这会向`users`表中添加一行新数据，包含`user_id`、`email`、`city`和`registration_date`的指定值。值的顺序必须与指定列的顺序匹配。

#### 修改数据：`UPDATE`语句

要更改表中的现有数据，使用`UPDATE`语句。

- **语法：** 你指定表、要更改的列（`SET`），以及重要的，要更改哪些行（`WHERE`）。

  ```sql
  UPDATE users
  SET email = 'updated.email@example.com'
  WHERE user_id = 101;
  ```

  此命令会找到`user_id`为101的行，并更改该特定行中`email`列的值。

  **重要：** `UPDATE`语句务必使用`WHERE`子句。如果省略它，你将意外地更新表中**每一行**的指定列，这通常不是你的意图，并可能导致严重的数据损坏。

#### 删除数据：`DELETE FROM`语句

要从表中删除整行，使用`DELETE FROM`语句。

- **语法：** 你指定表以及使用`WHERE`子句要删除的行。

  ```sql
  DELETE FROM users
  WHERE user_id = 101;
  ```

  此命令会删除`user_id`为101的行（或多行）。

  **重要：** 与`UPDATE`类似，使用`DELETE`时务必谨慎并使用`WHERE`子句。省略`WHERE`子句将删除表中的**所有行**。在执行`DELETE`语句之前，请仔细检查你的`WHERE`条件。

### SQL在数据工程工作流中的应用

这些基本的SQL数据操作命令是数据工程师的常用工具：

- **数据验证：** 将数据加载到暂存表后，你将使用带有`WHERE`子句的`SELECT`查询来检查空值、不正确的格式或异常值。
- **简单转换：** 有时，可以直接在数据库中使用`UPDATE`进行快速修正或标准化。
- **填充表：** 你可以使用`INSERT`语句来填充小型维度表或查找表。
- **数据清洗：** `DELETE`可用于删除重复记录或在验证后被识别为错误的行。
- **ETL/ELT逻辑：** 这些SQL命令通常构成大型数据管道脚本的核心部分，用于提取数据（`SELECT`）、可能在数据库内转换数据（`UPDATE`）、将数据加载到其他位置，或删除已处理的暂存数据（`DELETE`）。

虽然本入门内容涵盖了基础知识，但SQL提供了更多功能，包括合并来自多个表的数据（`JOIN`）、汇总数据（使用`COUNT`、`SUM`、`AVG`等聚合函数结合`GROUP BY`），以及进行复杂筛选。然而，掌握`SELECT`、`INSERT`、`UPDATE`和`DELETE`为与关系型数据库中存储的海量数据交互打下了坚实基础，使SQL成为你数据工程工具包中不可或缺的工具。

## 参考资料

- [PostgreSQL 16.1 Documentation: SQL Commands](https://www.postgresql.org/docs/current/sql-commands.html) — PostgreSQL Global Development Group (2023)
  提供PostgreSQL生态系统中所有SQL命令的官方规范和详细用法，包括DML操作。
- [Learning SQL, 3rd Edition](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本广受推荐的SQL初学者书籍，以实用的方式涵盖了SQL数据操作和查询的基础知识。
- [Database System Concepts, 7th Edition](http://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本经典的学术教材，提供了数据库系统、关系模型以及SQL理论基础的理解。
