---
course: "sql-for-data-science"
chapter: "intro-databases-sql"
lesson: "what-is-relational-database"
sourceId: 1545
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-1-intro-databases-sql/what-is-relational-database"
title: "什么是关系型数据库？"
description: "对关系型数据库、其结构和使用原因的简要说明。"
order: 1
plots: []
sourceHash: "04b3fb0996bc30cdee8e55f853d6153eb2fae3fbfac3fab36249365492730643"
sourceCorrections: []
---

设想你有一大批信息，可能是关于客户及其订单、学生及其课程，或产品及其库存量的。你如何储存这些信息，使其条理清晰、易于查找且可靠呢？你或许会使用电子表格，但随着数据量增大且关联性增强，管理起来会变得复杂且容易出错。这就是数据库的作用所在，尤其是其中一种非常常见的类型，称为*关系型数据库*。

数据库本质上就是一组有组织的结构化信息或数据集合，通常以电子形式存储在计算机系统中。它能让你高效地储存、获取、修改和管理这些数据。

那么，是什么让数据库“关系型”呢？这个术语源于数据组织的方式。在关系型数据库中，信息被组织成**表**。你可以把表想象成一个电子表格网格：它有列（表示不同的属性或信息片段，比如 `姓名` 或 `价格`）和行（表示单个记录或项目，比如特定的客户或产品）。

“关系型”部分是其最重要的特点。这表示这些表并非孤立的数据块。相反，它们可以基于共享的信息片段相互**关联**或**连接**。这种结构使你能够逻辑地连接不同类型的数据。

例如，你可能有一个表存储客户信息，另一个表存储订单信息。

- **客户表：** 包含 `CustomerID`、`FirstName`、`LastName`、`Email` 等列。每行代表一个独立的客户。
- **订单表：** 包含 `OrderID`、`OrderDate`、`CustomerID`、`TotalAmount` 等列。每行代表一个特定的订单。

关系型数据库通过共同的列连接不同的数据表。例如，一个客户信息表可能包含 `CustomerID` 和客户姓名，而一个订单信息表也包含 `CustomerID` 和订单详情。当这两个表都包含 `CustomerID` 时，这个共同的列就成为了连接或关联它们的纽带。利用这个共享的 `CustomerID`，你可以轻松地找到特定客户下过的所有订单，或查找与某个订单相关的客户详细信息。

> 一个简单的图示，展示了 `客户` 表和 `订单` 表如何通过一个共同的 `CustomerID` 列关联起来。

这种关系型结构，基于埃德加·F·科德于1970年提出的关系模型的基本思想，具有以下几项优势：

1. **组织清晰：** 数据被逻辑地分成代表不同概念（如客户、订单、产品）的表，使结构易于理解。
2. **数据完整性：** 关系型数据库允许你定义规则来维持数据的准确性和一致性。例如，你可以确保 `OrderID` 始终唯一，或者 `Orders` 表中的每个 `CustomerID` 确实存在于 `Customers` 表中。
3. **减少冗余：** 无需为客户的每个订单重复所有客户详细信息（姓名、地址等），你只需将客户信息在 `Customers` 表中存储一次，然后在 `Orders` 表中使用 `CustomerID` 进行引用。这节省了空间，并使更新更便捷（如果客户更改了电子邮件，你只需在一个地方更新它）。
4. **数据获取灵活：** 这些关联允许你以强大的方式组合多个表的数据，以回答复杂的问题。我们将在课程后期介绍如何使用 SQL 来实现这一点。
5. **标准化：** 大多数关系型数据库使用结构化查询语言 (SQL) 作为定义、操作和查询数据的标准方式。学习 SQL 将为你提供一项可在许多不同数据库系统（如 PostgreSQL、MySQL、SQL Server、Oracle）之间迁移的技能。

总之，关系型数据库是一种将数据存储在结构化表格中，并可基于共同信息相互关联的系统。这种模型为数据管理提供了一个有组织、高效且可靠的依据，这对于数据科学中的许多任务都十分必要，从基本报告到构建机器学习 (machine learning)模型。在接下来的章节中，我们将更详细地查看这些表的组成部分：列、行以及它们所包含的数据类型。

## 参考资料

- [A Relational Model of Data for Large Shared Data Banks](http://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf) — Edgar F. Codd (1970)
  Journal: Communications of the ACM; Publisher: ACM; Volume: 13; Pages: 377-387; DOI: [10.1145/362384.362685](https://doi.org/10.1145/362384.362685)
  介绍数据库管理关系模型的原始学术论文。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本广泛使用的学术教科书，全面介绍了数据库系统，包括关系数据库理论和设计原则。
- [Learning SQL](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本面向初学者的实用指南，用于学习结构化查询语言 (SQL)，这是与关系数据库交互的标准语言。
