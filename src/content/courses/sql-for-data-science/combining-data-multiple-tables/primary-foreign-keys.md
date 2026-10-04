---
course: "sql-for-data-science"
chapter: "combining-data-multiple-tables"
lesson: "primary-foreign-keys"
sourceId: 1626
sourceUrl: "https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/primary-foreign-keys"
title: "理解主键和外键"
description: "学习主键和外键，以及它们在关联表中的连接作用。"
order: 2
plots: []
sourceHash: "27e30c11c3201bf539ef10627f2a31956c76a1588fe34e58c076b310c1301368"
sourceCorrections: []
---

如章引言所述，关系型数据库中的数据通常分布在多张表中，以保持结构清晰并减少冗余。设想一下，如果试图将顾客的全部信息*以及*他们所有的订单都存储在一张庞大的表中。这会很快变得难以管理且重复繁琐。相对而言，我们通常会有独立的表，例如一张用于顾客详情，另一张用于订单信息。

但这些独立的表之间如何关联呢？数据库如何判断哪笔订单属于哪位顾客？这正是主键和外键的作用所在。它们充当连接器，用于建立和维护关联表之间的联系。

## 主键：唯一标识符

设想一张表是记录的集合，如同一个在线商店的全部顾客列表。`Customers`表中的每一行都代表一位独特的顾客。为了可靠地区分不同顾客，即使他们碰巧同名，我们也需要为每行设置一个唯一标识。这个唯一标识即是**主键**。

主键是表中唯一标识每一行的一个列（或有时是多个列的组合）。

主键列的主要特点：

1. **唯一性：** 主键列中的每个值必须唯一。任何两行不能拥有相同的主键值。
2. **非空性：** 主键列通常不能包含`NULL`值。每行都必须拥有一个有效的标识。

考虑一个简单的`Customers`表：

| 顾客ID | 名 | 姓 | 电子邮件 |
| --- | --- | --- | --- |
| 101 | Alice | Smith | [alice.s@example.com](mailto:alice.s@example.com) |
| 102 | Bob | Johnson | [b.johnson@example.com](mailto:b.johnson@example.com) |
| 103 | Charlie | Davis | [charlie.d@example.com](mailto:charlie.d@example.com) |
| 104 | Alice | Brown | [alice.b@example.com](mailto:alice.b@example.com) |

在此表中，`customer_id`作为主键。每位顾客都拥有一个独特的`customer_id`（101, 102, 103, 104），这确保我们能够准确定位任何特定的顾客记录。即使有两位顾客名为“Alice”，他们独特的`customer_id`值（101和104）也使我们能够区分他们。电话号码或电子邮件地址可能看似唯一，但它们可能发生变化或并非适用于所有顾客，因此像`customer_id`这样专用的ID列是主键的更可靠选择。

## 外键：连接表

现在，我们引入一个`Orders`表来存储顾客购买信息：

| 订单ID | 订单日期 | 顾客ID | 总金额 |
| --- | --- | --- | --- |
| 5001 | 2023-10-26 | 101 | 45.50 |
| 5002 | 2023-10-26 | 103 | 120.00 |
| 5003 | 2023-10-27 | 101 | 15.75 |
| 5004 | 2023-10-28 | 104 | 88.20 |

`Orders`表也有其自己的主键`order_id`，以唯一标识每笔订单。但请注意`customer_id`列。我们如何知道哪位顾客下了订单5001？我们查看该行中的`customer_id`值（101），并在`Customers`表中找到匹配的`customer_id`（101）。这告诉我们订单是由Alice Smith下的。

`Orders`表中的`customer_id`列是**外键**。

外键是某个表中的一个列（或多个列的组合），其值对应于另一个表中的主键值。它充当交叉引用，将包含外键的表（本例中为`Orders`表）中的行链接到包含相应主键的表（`Customers`表）中的行。

主键告诉我们“这是*此表中*此行的唯一ID。”外键告诉我们“此值将此行与*另一个表中*的特定行关联起来。”

## 强制关系：参照完整性

外键不仅仅是标签；数据库系统常使用它们来强制执行**参照完整性**。这表示数据库确保表之间的关系保持一致。例如，在`Orders`和`Customers`表之间，`customer_id`列上正确定义外键关系后：

- 你通常不能向`Orders`表插入一个`customer_id`在`Customers`表中不存在的订单。这可避免出现属于不存在顾客的“孤立”订单。
- 根据数据库设置，规则可能会阻止你从`Customers`表删除一个在`Orders`表中仍有关联订单的顾客，或者它可能会自动处理这些关联订单（例如，通过一并删除它们或在允许的情况下将其`customer_id`设为`NULL`）。

这些约束有助于维护关联表中数据的逻辑一致性。

> 此图表显示了`Customers`和`Orders`表之间的关系。`Customers`表中的`customer_id`（PK - 主键）被`Orders`表中的`customer_id`（FK - 外键）引用，从而在关联记录之间建立了联系。

在合并数据之前，了解主键和外键非常重要。它们提供逻辑结构，使我们能够准确地合并来自不同表的信息，这正是`SQL JOIN`操作的设计目的。在接下来的章节中，我们将了解如何在`JOIN`子句中使用这些键来获取组合数据集。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本基础教材，全面涵盖关系数据库理论，包括键和参照完整性的详细解释。
- [CREATE TABLE (Constraints)](https://www.postgresql.org/docs/current/sql-createtable.html#SQL-CREATETABLE-CONSTRAINTS) — PostgreSQL Global Development Group (2024)
  官方文档，详细说明在PostgreSQL中定义主键和外键约束的SQL语法和行为。
- [Learning SQL: Master SQL Fundamentals](https://books.google.com/books/about/Learning_SQL_Master_SQL_Fundamentals.html?id=R61M6rXwL3MC) — Alan Beaulieu (2009)
  Publisher: O'Reilly Media; Pages: 336
  一本实用的SQL基础指南，清晰解释了如何在各种SQL数据库中实现和使用主键和外键，并提供示例。
