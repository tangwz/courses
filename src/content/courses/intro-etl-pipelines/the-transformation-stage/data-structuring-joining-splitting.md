---
course: "intro-etl-pipelines"
chapter: "the-transformation-stage"
lesson: "data-structuring-joining-splitting"
sourceId: 5543
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-3-the-transformation-stage/data-structuring-joining-splitting"
title: "数据结构化：数据连接与拆分"
description: "学习从多个来源组合数据或拆分数据集的基本方法。"
order: 7
plots: []
sourceHash: "67c896fc83b137c98441ee9e24e842253f4282d04541fc2729b5e41dfbcf9144"
sourceCorrections: []
---

通常，您提取的数据并非完美地集中在一处。关于一个单独实体的信息，例如客户，可能部分存在于一个数据集中（例如，基本资料），部分存在于另一个数据集中（例如，他们的订单历史）。同样地，单个数据集可能包含过多不同类型的信息，使其难以管理。数据结构化技术，特别是连接和拆分，有助于在转换阶段重塑您的数据，以便为分析或加载到目标系统创建更具逻辑性和实用性的结构。

### 组合相关数据：连接操作

连接是将两个或更多表或数据流中的行，根据它们之间的一个关联列进行组合的过程。可以把它想成是匹配拼图碎片。如果您有一个客户列表和一个单独的订单列表，您可以使用一个共同的标识符（例如，`CustomerID`）将它们“连接”起来，以显示哪个客户下了哪个订单。

用于匹配的列通常被称为**连接键**。为了使连接有效运行，这些键需要存在于您希望组合的数据集中。常见的例子包括唯一标识符（例如`UserID`、`ProductID`、`OrderID`），或者有时是字段的组合。

有几种连接数据的方式，取决于您希望保留哪些信息：

1. **内连接：** 这是最常见的类型。内连接只返回两个表中都存在连接的行。如果一个客户存在于客户列表中但从未下过订单，他们将不会出现在客户和订单之间内连接的结果中。

   > 内连接根据匹配的`CustomerID`组合客户和订单。查理（ID 3）被排除，因为他没有订单。
2. **左连接（或左外连接）：** 这种连接返回“左”表（连接操作中提到的第一个表）中的*所有*行，以及“右”表（第二个表）中匹配的行。如果左表中的某行在右表中没有匹配项，则右表中的列将包含`NULL`或空值。当您希望保留一个主表的所有记录并添加相关信息（如果存在）时，这很有用。

   > 左连接保留所有客户（左表），并在`CustomerID`匹配时添加订单详情。查理（ID 3）被包含在内，但`OrderID`和`Product`为`NULL`，因为他没有匹配的订单。
3. **右连接（或右外连接）：** 这与左连接是镜像关系。它返回“右”表中的所有行以及“左”表中匹配的行。右表中在左表没有匹配项的行，其左表对应列将包含`NULL`值。
4. **全外连接：** 这种连接返回左右表中任意一方存在匹配的所有行。它本质上结合了左连接和右连接的结果。如果一个表中的某行在另一个表中没有匹配项，则非匹配表中的列将填充`NULL`值。

选择正确的连接类型完全取决于您希望回答的问题或目标系统所需的结构。

### 拆分数据：拆分

拆分是将一个大型表或数据集分解为两个或更多更小、更集中的表的过程。这通常是为了组织、效率，或者符合一个特定的数据库设计原则，称为**规范化**。

规范化的目的是减少数据冗余（避免多次存储相同的信息）并提高数据完整性。

以下是拆分有用的常见情况：

1. **分离重复组：** 想象一个数据集，其中每行包含订单ID、客户详细信息（姓名、地址、电子邮件）和产品详细信息（产品ID、名称、价格）。如果一个客户下达多个订单，他们的姓名、地址和电子邮件将在每一行中重复。将其拆分为一个`Orders`表（订单ID、客户ID、订单日期）和一个`Customers`表（客户ID、姓名、地址、电子邮件）可以消除这种重复。这些表可以使用`CustomerID`进行关联。
2. **分解复杂字段：** 有时，单个列包含多条信息。例如，单个`Address`列可能包含“123 Main St, Anytown, CA 94000”。将其拆分为单独的`StreetAddress`、`City`、`State`和`ZipCode`列，使得数据更易于查询和分析（例如，查找特定州的所有客户）。
3. **分离不同实体：** 一个宽表可能混合了不同实体的信息。例如，一个产品表可能包含供应商信息（供应商名称、联系方式）。将其拆分为一个`Products`表和一个`Suppliers`表，并通过`SupplierID`关联，可以使数据模型更清晰、更易于理解。每个表都专注于一个主题。

#### 示例：拆分客户和地址数据

假设您从如下数据开始：

**初始数据：**

| 客户ID | 姓名 | 完整地址 |
| --- | --- | --- |
| 1 | Alice | 10 Pine St, Metro, NY |
| 2 | Bob | 25 Oak Ave, Gotham, NJ |
| 3 | Alice | 10 Pine St, Metro, NY |

请注意爱丽丝的地址重复了。我们可以将其拆分：

**客户表（拆分后）：**

| 客户ID | 姓名 | 地址ID |
| --- | --- | --- |
| 1 | Alice | 101 |
| 2 | Bob | 102 |
| 3 | Alice | 101 |

**地址表（拆分后）：**

| 地址ID | 街道 | 城市 | 州 |
| --- | --- | --- | --- |
| 101 | 10 Pine St | Metro | NY |
| 102 | 25 Oak Ave | Gotham | NJ |

现在，“10 Pine St, Metro, NY”这个地址只在`Addresses`表中存储了一次，由`AddressID` 101标识。`Customers`表使用此`AddressID`将客户与其地址关联起来，从而减少了冗余。如果爱丽丝的地址发生变化，您只需在一个地方更新它。

连接和拆分都是基本的数据转换技术，用于将数据重塑为干净、高效且适合加载到目标系统或进行后续分析步骤的格式。它们使您能够组合不同的信息源，或将复杂的数据集分解为更易于管理和更具逻辑性的组成部分。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本数据库系统的基础教材，涵盖关系代数、SQL和数据库规范化，这些对于理解数据结构至关重要。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons
  数据仓库的权威指南，提供了在ETL流程和维度建模中，数据转换技术（如连接和拆分）的应用背景。
- [7.2. Joins Between Tables](https://www.postgresql.org/docs/current/queries-tableexpr.html) — PostgreSQL Global Development Group (2025)
  Publisher: PostgreSQL Global Development Group
  官方文档解释了各种SQL连接类型，包含实际语法和示例，与数据合并直接相关。
- [Learning SQL](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057604/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 380 pages
  一本学习SQL的实用书籍，涵盖连接操作和数据库设计原则，有助于数据结构的实际应用。
