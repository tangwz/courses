---
course: "sql-for-data-science"
chapter: "intro-databases-sql"
lesson: "tables-columns-rows"
sourceId: 1547
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-1-intro-databases-sql/tables-columns-rows"
title: "表、列和行"
description: "了解数据库表的基本组成部分：列（数据项）和行（记录）。"
order: 2
plots: ["plots/1547-0.json"]
sourceHash: "4fc9ad3f1a14e7629dfd9365401d4f667383a4269ef5bf93943ce977e55ff8bd"
sourceCorrections: []
---

我们已经知道，关系型数据库旨在以有序的方式存储数据。但这种有序性具体是什么样的呢？每个关系型数据库的核心都是**表**。可以把表看作数据的主要容器，它类似于电子表格程序中的单个工作表，但具有更严格的结构和规则。每个表通常保存有关某一特定类别项目的数据，例如客户、产品、订单或实验结果。

## 表内数据排列

假设您想保存客户数据。您不会将所有数据随意放入一个无序的文件中。相反，您会使用一个表，例如命名为`Customers`。在这个表中，数据按列和行定义的网格形式排列。

### 列：定义数据项

**列**在表中垂直排列。每列代表关于表所描述实体的特定数据项或信息片段。对于我们的`Customers`表，列可能包括`CustomerID`、`FirstName`、`LastName`、`Email`和`SignupDate`。

列的一个重要方面是它们规定了所包含数据的*类型*。`CustomerID`列可能保存整数，而`FirstName`和`LastName`会保存文本（字符串），`SignupDate`则保存日期。输入到特定列中的每条数据都必须符合该列定义的数据类型和含义。这保证了数据的一致性，并使其在分析时可信。您可以将列看作您希望追踪的“字段”或“属性”。

### 行：表示独立记录

**行**在表中水平排列。每行代表表所描述实体的单一、完整记录或一个实例。在我们的`Customers`表中，每行都与一个特定客户相对应。它包含了该客户在所有已定义列中的实际数据值。例如，一行可能包含`101`、`'Alice'`、`'Smith'`、`'alice.s@email.com'`和`'2023-01-15'`。另一行则代表一个具有其特有值集合的不同客户。行通常被称为“记录”，在更正式的数据库术语中有时也称为“元组”。

## 可视化例子

让我们看一个小的`Customers`表：



![客户表示例](plots/1547-0.json)



> 这个表展示了四条客户记录（行）。每条记录都包含五个已规定数据项（列）的值：CustomerID、FirstName、LastName、Email和SignupDate。

简单来说，表提供结构（列规定数据项），行提供独立数据条目（记录）。这种明确的区分和排列使得使用SQL高效保存、管理和获取特定数据成为可能，我们很快就会开始学习SQL。理解这种由列和行构成的表结构，对于编写好的查询语句非常重要。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本备受推崇的综合性教科书，系统地介绍了关系数据库的基础知识，清晰定义了表、属性（列）和元组（行）。
- [Learning SQL: Master the Fundamentals of SQL and Relational Databases](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380
  一本为初学者准备的实用易懂的书籍，在深入SQL查询之前，解释了关系数据库结构的核心概念，包括表、列和行。
