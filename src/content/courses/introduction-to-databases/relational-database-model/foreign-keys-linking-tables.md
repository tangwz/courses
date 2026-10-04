---
course: "introduction-to-databases"
chapter: "relational-database-model"
lesson: "foreign-keys-linking-tables"
sourceId: 1577
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-2-relational-database-model/foreign-keys-linking-tables"
title: "外键：连接表"
description: "说明外键的作用，以及它们如何创建和维护相关表之间的联系。"
order: 6
plots: []
sourceHash: "0827ef4364fbd217423ddcc9f5b11595633659b06ad612e9523d6d05484dd81f"
sourceCorrections: []
---

关系数据库中的表通常并非独立存在；一个表中的信息常常与另一个表中的信息直接相关。例如，`Orders`（订单）表需要知道每笔订单是由*哪个*顾客下的。为了建立和维护这些表之间的联系，关系数据库使用一种特定的机制：**外键**。

### 外键到底是什么？

可以把外键看作连接两个表的桥梁。从技术上讲，**外键**是一个表中的列或一组列，其值必须与*另一个*表的主键列中的值相匹配。包含外键的表通常称为*子表*或*引用表*，而被其引用的主键所在的表则称为*父表*或*被引用表*。

外键的主要作用有两方面：

1. **建立联系：** 它明确地将一个表中的行与另一个表中的特定行连接起来。
2. **维护数据一致性：** 它确保表之间的关系保持有效。这种特性称为**引用完整性**。

### 表的连接：一个例子

让我们再次使用常用示例：`Customers`（顾客）和 `Orders`（订单）。

假设我们有一个包含顾客详细信息的 `Customers` 表。每个顾客都需要一个唯一标识符，我们已经确定这是**主键**。我们称之为 `CustomerID`。

```
顾客表
+--------------+----------+-------+
| CustomerID   | 姓名     | 邮箱  |
+--------------+----------+-------+
| 101 (PK)     | Alice    | a@... |
| 102 (PK)     | Bob      | b@... |
| 103 (PK)     | Charlie  | c@... |
+--------------+----------+-------+
 (PK = 主键)
```

现在，我们有一个 `Orders`（订单）表来存储订单信息。每笔订单也需要一个唯一标识符（`OrderID` 作为主键）。重要的是，每笔订单都必须属于 `Customers` 表中的一位顾客。我们如何将订单与下订单的顾客联系起来呢？我们在 `Orders` 表中添加一个 `CustomerID` 列。`Orders` 表中的这个 `CustomerID` 列充当**外键**。

```
订单表
+------------+----------+----------------+
| OrderID    | 产品     | CustomerID     |
+------------+----------+----------------+
| 5001 (PK)  | Laptop   | 101 (FK)       |
| 5002 (PK)  | Keyboard | 102 (FK)       |
| 5003 (PK)  | Monitor  | 101 (FK)       |
| 5004 (PK)  | Mouse    | 103 (FK)       |
+------------+----------+----------------+
 (PK = 主键, FK = 外键，引用 Customers.CustomerID)
```

在 `Orders` 表中：

- `OrderID` 是主键，唯一标识每笔订单。
- `CustomerID` 是外键。此列中的值（101、102、103）必须与 `Customers` 表中已有的 `CustomerID` 值相对应。

这个外键创建了一个直接联系：订单 5001 属于顾客 101 (Alice)，订单 5002 属于顾客 102 (Bob)，依此类推。

这是该关系的一个图示：

> `Orders` 表中的 `CustomerID` 列是一个外键，它与 `Customers` 表中的 `CustomerID` 主键关联，从而建立了订单与顾客之间的联系。

### 引用完整性：保持联系的有效性

外键的第二个作用是维护**引用完整性**。这是确保表之间关系一致性的重要原则。它主要指的是：

1. 您不能向子表（`Orders`）添加记录，除非外键值（`CustomerID`）与父表（`Customers`）中现有的主键值相符。例如，如果 `Customers` 表中没有 `CustomerID` 为 104 的顾客，您就无法添加一个 `CustomerID` 为 104 的订单。数据库会拒绝此操作，以防止出现没有关联到有效顾客的“孤立”订单记录。
2. 当尝试删除或更新父表（`Customers`）中被子表（`Orders`）记录引用的记录时，通常会应用一些规则。例如，数据库可能会阻止您从 `Customers` 表中删除顾客 101，因为订单 5001 和 5003 引用了该顾客。允许删除将导致这些订单指向一个不存在的顾客，从而破坏了关系的完整性。数据库系统提供了不同的方法来处理这些情况（例如限制删除、同时删除相关订单，或者将外键设为 NULL），但目标始终是保持数据一致性。

外键是维系关系数据库结构的纽带。它们使得您可以将复杂信息分解为逻辑、易于管理的表，同时维护它们之间有意义且稳定的联系。如果没有外键，确保相关表之间的数据一致性将非常困难且容易出错。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education
  一本全面且广泛使用的教材，涵盖了关系型数据库的理论基础，包括主键、外键和参照完整性强制执行的详细解释。
- [PostgreSQL Documentation: 5.3. Foreign Keys](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-FK) — The PostgreSQL Global Development Group (2024)
  PostgreSQL 官方文档，详细介绍了外键约束的实际实现和语法，包括用于维护数据一致性的各种参照动作（例如，ON DELETE，ON UPDATE）。
