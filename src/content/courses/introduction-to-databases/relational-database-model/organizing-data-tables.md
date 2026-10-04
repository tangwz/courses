---
course: "introduction-to-databases"
chapter: "relational-database-model"
lesson: "organizing-data-tables"
sourceId: 1562
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-2-relational-database-model/organizing-data-tables"
title: "表格中的数据组织"
description: "了解关系数据库的基本结构：由行（记录）和列（属性）组成的表。"
order: 1
plots: []
sourceHash: "9ee2214804a8040aa9b2785a1860a24e2b050a779a5e61e4fb71e2d41a642196"
sourceCorrections: []
---

在确定了数据存储有序性的需求之后，我们现在来看关系模型如何实现这一点。关系数据库中最基本的思想是**表**。可以把表看作数据的主要载体，其外观类似于电子表格的网格，但具有更强的结构和规则。

关系模型提供了一种组织数据的结构化方法。在关系数据库中，**表**是最基本的概念。表是数据的主要容器，外观类似于电子表格，但具有更多的结构和规则。每个表旨在存储关于特定类型实体的信息，例如客户、产品或订单。例如，你可能有一个`Customers`表来存储购买你产品的人的信息，以及一个独立的`Products`表来存储你销售的物品。

## 结构：列和行

表将数据组织成一个由列和行构成的二维结构。

### 列（属性/字段）

列在表中垂直排列。每列代表关于表所描述实体的特定属性或信息片段。例如，在一个`Customers`表中，你可能拥有以下列：

- `CustomerID`
- `FirstName`
- `LastName`
- `Email`
- `SignupDate`

表中的每列都有一个唯一名称（例如，`FirstName`），并且被定义为保存特定*类型*的数据（如文本、数字或日期）。数据类型对于保证数据完整性很重要，并将会在下一节讨论。列名集合确定了表的结构。

### 行（记录/元组）

行在表中水平排列。每行代表表所描述实体的单个实例或记录。在我们的`Customers`表中，每行将包含一个特定客户的信息。例如，一行可能保存爱丽丝·史密斯的`CustomerID`、`FirstName`、`LastName`、`Email`和`SignupDate`，而另一行则保存鲍勃·琼斯的相同信息片段。

表中的每一行都遵循由列定义的结构。它包含每个列的值，即使该值有时被指定为未知或`NULL`（一个我们稍后会遇到的特殊数据库标记 (token)）。

## 表格可视化

我们来查看一个简单的`Products`表结构。

> `Products`表的一个基本示意图。列（`ProductID`、`ProductName`、`Category`、`Price`）定义了结构，每行代表一个具体的产品记录。

在这个例子中：

- 该表名为`Products`。
- 它有四列：`ProductID`、`ProductName`、`Category`和`Price`。这些列定义了每个产品所存储的属性。
- 它当前显示三行，每行代表一个具有相应详细信息的不同产品。

尽管网格格式类似于电子表格，但数据库表受关系模型和数据库管理系统（DBMS）定义的更严格规则的约束。这些规则保证数据一致性，实现高效查询，并促成不同表之间的关联，我们将在稍后查看。

理解表、列和行的这种基本结构非常重要。它是所有关系数据库操作所依赖的结构。在接下来的章节中，我们将更细致地查看列和数据类型，引入用于唯一标识行的键的理念，并了解表如何关联起来。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education
  一本广泛用于数据库入门课程的教科书，全面涵盖了关系数据库原理，包括表结构、列和行。
- [PostgreSQL 16.2 Documentation: 5.1. Tables](https://www.postgresql.org/docs/16/ddl-tables.html) — The PostgreSQL Global Development Group (2024)
  Publisher: The PostgreSQL Global Development Group
  官方文档，在流行的开源关系数据库系统背景下，对表定义、列和行提供了实用的解释。
