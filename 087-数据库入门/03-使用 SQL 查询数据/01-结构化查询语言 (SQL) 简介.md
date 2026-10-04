# 结构化查询语言 (SQL) 简介

来源：[原文](https://apxml.com/zh/courses/introduction-to-databases/chapter-3-querying-data-sql/introduction-structured-query-language-sql)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您已经了解了关系数据库如何使用主键和外键将信息组织成整齐的行和列的表格，以建立数据结构和关联。这是一个很好的起点。但您如何真正地与这些数据*交互*呢？您如何向数据库提问、添加新信息或纠正错误呢？

答案就是**结构化查询语言**，它通常被称为 **SQL**（发音常为“sequel”或直接读作 S-Q-L）。

可以把 SQL 看作是您用来与关系数据库管理系统 (DBMS) 通信的专用语言。就像您使用英语或西班牙语与他人交谈一样，您使用 SQL 与数据库对话。这是告诉数据库如何处理其所存数据的标准方式。

### 什么是 SQL？

SQL 是 **结构化查询语言** 的缩写。我们来具体看看它的含义：

- **结构化 (Structured)：** SQL 命令遵循特定的、一致的语法和文法。这种结构确保您的指令对数据库系统而言清晰明确。您必须正确地构建请求，数据库才能理解它们。
- **查询 (Query)：** 尽管“查询”通常意味着提问（检索数据），但 SQL 的功能不止于此。它用于查询数据，也用于插入新数据、更新现有数据和删除数据。它还可以用于管理数据库本身的数据结构，例如创建或修改表（尽管我们首先将侧重于数据交互）。
- **语言 (Language)：** 它是一种完整的语言，专门用于管理和处理关系数据库中的数据。

想象您的数据库是一个庞大而组织严密的仓库，而 DBMS 是仓库经理。SQL 就是您交给经理的一系列指令：“找出上周所有标记 (token)为‘客户订单’的箱子”、“将此新物品添加到 3B 货架上”或“更新物品 X 的位置”。

### 标准及其变体

SQL 最早于 20 世纪 70 年代开发，此后已成为美国国家标准学会 (ANSI) 和国际标准化组织 (ISO) 认可的官方标准。这种标准化非常有益，因为它意味着核心命令在不同的关系数据库系统中工作方式相似。

然而，尽管核心语言是标准化的，但大多数特定的数据库系统（如 PostgreSQL、MySQL、Microsoft SQL Server、Oracle Database、SQLite）都会在标准 SQL 语法中添加自己的专有扩展和变体。可以将其视为口语方言——来自不同地区的人可能会使用略微不同的词语或表达方式，但核心语言仍然可以理解。对于本课程中涉及的基本操作（`SELECT`、`INSERT`、`UPDATE`、`DELETE`），语法在大多数常用数据库中都非常一致。

### SQL 操作类型

SQL 命令可以根据其功能进行大致分类。虽然您现在不需要记住这些类别，但查看它们有助于您理解 SQL 的应用范围：

1. **数据查询语言 (DQL)：** 用于检索数据。这里的主要命令是 `SELECT`，您将大量使用它来向数据库请求特定信息。
2. **数据操作语言 (DML)：** 用于管理现有表中的数据。这包括添加新数据（`INSERT`）、修改现有数据（`UPDATE`）和删除数据（`DELETE`）。
3. **数据定义语言 (DDL)：** 用于定义和管理数据库本身的数据结构。`CREATE TABLE`、`ALTER TABLE` 和 `DROP TABLE` 等命令属于此类别。这些命令构建数据的“容器”。
4. **数据控制语言 (DCL)：** 用于管理权限和访问控制。`GRANT`（授予权限）和 `REVOKE`（撤销权限）等命令属于此处。

在本章中，我们的主要侧重将是 **DQL**（使用 `SELECT` 获取数据）和 **DML**（使用 `INSERT`、`UPDATE`、`DELETE` 更改数据），因为这些是处理存储在表中信息的基本操作。

现在您已经了解了 SQL 及其用途，我们准备好开始学习实际的命令了。我们将从最常见的操作开始：使用 `SELECT` 语句检索数据。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education
  一本全面的学术教科书，为数据库系统提供了坚实的基础，并详细介绍了SQL原理和应用。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057632/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本高度实用的SQL基础书，适合初学者到中级用户使用各种关系型数据库系统。
- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/index.html) — The PostgreSQL Global Development Group (2025)
  Publisher: The PostgreSQL Global Development Group
  领先的开源关系型数据库的官方文档，详细说明了SQL命令及其在PostgreSQL中的具体实现。

---

[上一节](../02-%E5%85%B3%E7%B3%BB%E5%9E%8B%E6%95%B0%E6%8D%AE%E5%BA%93%E6%A8%A1%E5%9E%8B/07-%E8%A7%84%E8%8C%83%E5%8C%96%E5%8E%9F%E5%88%99%E7%AE%80%E4%BB%8B.md) · [下一节](02-%60SELECT%60%20%E8%AF%AD%E5%8F%A5%EF%BC%9A%E8%8E%B7%E5%8F%96%E6%95%B0%E6%8D%AE.md)
