# 练习编写基础 SQL 查询

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/practice-basic-sql-queries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

练习编写基本的 SQL 查询，使用与数据交互的基本语句，例如 `SELECT`、`INSERT`、`UPDATE`、`DELETE`，以及 `WHERE` 和 `ORDER BY`。通过示例练习是巩固理解并建立编写自己查询信心的最佳方式。

### 设置练习环境

对于这些练习，假设我们有一个小型图书馆或书店的简单数据库。它包含两个表：`Authors` 和 `Books`。

**Authors 表：**

此表存储作者信息。

| author\_id | first\_name | last\_name |
| --- | --- | --- |
| 1 | Jane | Austen |
| 2 | George | Orwell |
| 3 | Mary | Shelley |
| 4 | Charles | Dickens |

- `author_id`：唯一标识每位作者的数字（主键）。
- `first_name`：作者的名。
- `last_name`：作者的姓。

**Books 表：**

此表存储书籍信息，通过 `author_id` 将每本书与作者关联起来。

| book\_id | title | publication\_year | author\_id |
| --- | --- | --- | --- |
| 101 | Pride and Prejudice | 1813 | 1 |
| 102 | Emma | 1815 | 1 |
| 103 | 1984 | 1949 | 2 |
| 104 | Animal Farm | 1945 | 2 |
| 105 | Frankenstein | 1818 | 3 |
| 106 | Oliver Twist | 1838 | 4 |
| 107 | Great Expectations | 1861 | 4 |

- `book_id`：唯一标识每本书的数字（主键）。
- `title`：书名。
- `publication_year`：图书出版年份。
- `author_id`：一个关联到 `Authors` 表中 `author_id` 的数字（外键）。

您目前无需自行创建这些表。只需使用此结构和数据作为编写下方查询的依据。如果您可以使用数据库工具（例如 DB Browser for SQLite、DBeaver 或像 SQL Fiddle 这样的在线 SQL 环境），您可以创建这些表并插入数据来运行查询。

### 练习

让我们完成一些使用 SQL 的常见任务。在查看提供的解决方案之前，请尝试自己编写查询。

**练习 1：检索所有作者**

任务：编写查询以从 `Authors` 表中检索所有作者的所有信息。

解决方案：

```sql
SELECT *
FROM Authors;
```

这是最基础的 `SELECT` 查询，使用 `*` 表示“所有列”。

**练习 2：检索特定书籍信息**

任务：编写查询以仅检索 `Books` 表中所有书籍的 `title` 和 `publication_year`。

解决方案：

```sql
SELECT title, publication_year
FROM Books;
```

在这里，我们指定了我们想要的确切列，而不是使用 `*`。

**练习 3：按作者筛选书籍**

任务：编写查询以查找 `author_id` 为 2（乔治·奥威尔）的作者所写的所有书籍。检索这些书籍的所有列。

解决方案：

```sql
SELECT *
FROM Books
WHERE author_id = 2;
```

`WHERE` 子句筛选行，仅返回 `author_id` 列与值 2 匹配的行。

**练习 4：按出版年份筛选书籍**

任务：编写查询以查找 1900 年 *之后* 出版的所有书籍。检索它们的 `title` 和 `publication_year`。

解决方案：

```sql
SELECT title, publication_year
FROM Books
WHERE publication_year > 1900;
```

`WHERE` 子句使用大于运算符（`>`）进行数值比较。

**练习 5：按书名排序书籍**

任务：编写查询以检索所有书籍的 `title` 和 `author_id`，并按 `title` 字母顺序排序。

解决方案：

```sql
SELECT title, author_id
FROM Books
ORDER BY title ASC;
```

`ORDER BY title` 根据 `title` 列对结果进行排序。`ASC` 指定升序（A-Z），这通常是默认值，但明确指出是个好习惯。要按降序（Z-A）排序，您可以使用 `DESC`。

**练习 6：添加新作者**

任务：编写查询以向 `Authors` 表添加一位新作者“列夫·托尔斯泰”。分配其 `author_id` 为 5。

解决方案：

```sql
INSERT INTO Authors (author_id, first_name, last_name)
VALUES (5, 'Leo', 'Tolstoy');
```

`INSERT INTO` 语句添加新行。我们指定表、要填充的列以及使用 `VALUES` 子句对应的数值。

**练习 7：添加新书籍**

任务：编写查询以将列夫·托尔斯泰的图书《战争与和平》（1869 年出版）添加到 `Books` 表。假设其 `book_id` 为 108，列夫·托尔斯泰的 `author_id` 为 5。

解决方案：

```sql
INSERT INTO Books (book_id, title, publication_year, author_id)
VALUES (108, 'War and Peace', 1869, 5);
```

类似于添加作者，我们向 `Books` 表中插入一个新行。

**练习 8：更新书名**

任务：原来 `book_id` 为 105 的书名应该是“弗兰肯斯坦；或现代普罗米修斯”。编写查询以更新此书的书名。

解决方案：

```sql
UPDATE Books
SET title = 'Frankenstein; or, The Modern Prometheus'
WHERE book_id = 105;
```

`UPDATE` 语句修改现有数据。`SET` 指定列及其新值。`WHERE` 子句在这里很重要；如果没有它，*所有* 书名都将被更改。除非您打算修改所有行，否则在使用 `UPDATE` 时务必使用 `WHERE`。

**练习 9：删除书籍**

任务：假设书籍《爱玛》（`book_id` = 102）从馆藏中移除。编写查询以从 `Books` 表中删除此书。

解决方案：

```sql
DELETE FROM Books
WHERE book_id = 102;
```

`DELETE FROM` 语句删除行。同样，`WHERE` 子句对于指定要删除 *哪些* 行很重要。省略 `WHERE` 将删除表中的 *所有* 行。使用 `DELETE` 时务必非常小心。

### 继续练习

这些练习涵盖了使用 SQL 的基础创建、读取、更新、删除（CRUD）操作。熟练掌握 SQL 的最佳方式是继续练习。

- 尝试使用不同条件修改 `WHERE` 子句（例如，查找特定年份 *之前* 出版的书籍，查找特定姓氏的作者）。
- 尝试对不同列使用 `ORDER BY`，并使用 `DESC` 进行降序排序。
- 如果您建立了练习数据库，请尝试添加更多作者和书籍，更新不同字段，并选择性地删除记录。

理解这些基本命令将为您处理关系型数据库打下坚实功底。随着您的进步，您将学习更高级的 SQL 功能，以进行更复杂的数据检索和操作。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Abraham Silberschatz, Henry F. Korth, S. Sudarshan (2020)
  Publisher: McGraw-Hill Education
  一本基础教材，全面涵盖数据库原理，包括SQL的详细解释和示例。
- [PostgreSQL 16 Documentation: Chapter 4. SQL Syntax](https://www.postgresql.org/docs/current/sql-syntax.html) — The PostgreSQL Global Development Group (2025)
  官方文档，提供了SQL语法和命令的权威参考，有助于理解标准SQL。
- [SQLZoo SQL Tutorial](https://sqlzoo.net/) — SQLZoo (2024)
  一个互动式在线平台，提供从基础到高级的SQL查询实践练习。

---

[上一节](07-DELETE%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E5%88%A0%E9%99%A4%E6%95%B0%E6%8D%AE.md) · [下一节](../04-NoSQL%20%E6%95%B0%E6%8D%AE%E5%BA%93%E6%A6%82%E8%A7%88/01-%E5%85%B3%E7%B3%BB%E5%9E%8B%E6%95%B0%E6%8D%AE%E5%BA%93%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
