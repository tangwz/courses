---
course: "introduction-to-databases"
chapter: "relational-database-model"
lesson: "rows-as-records"
sourceId: 1568
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-2-relational-database-model/rows-as-records"
title: "行作为记录"
description: "了解表中每行如何表示一个独立记录或实体。"
order: 3
plots: []
sourceHash: "d491bd132de720a13ca571eb9721c708d652a342ce1bd1e3e1b5a25ff1572dcf"
sourceCorrections: []
---

关系数据库将数据组织成表，这些表有列（或属性）定义了存储信息的*种类*，例如`FirstName`、`EmailAddress`或`ProductPrice`。行用于存储关于*某个*人、电子邮件或产品的具体信息。

可以把表看作是一个结构化的网格，类似于电子表格。如果列代表了信息的分类（例如在宠物表中是`Name`、`Species`、`Age`），那么每一**行**就代表了该分类集中的一个独立、完整条目或项。每一行为表中描述的某一个事物的特定列保存了具体的值。

例如，考虑一个简单的`Pets`表：

| PetID | Name | Species | Age |
| --- | --- | --- | --- |
| 1 | Fido | Dog | 5 |
| 2 | Whiskers | Cat | 3 |
| 3 | Buddy | Dog | 8 |

在这个表中：

- 第一行（`PetID` 1, `Name` Fido, `Species` Dog, `Age` 5）表示**一只特定的宠物**：狗狗Fido。
- 第二行表示**另一只特定的宠物**：猫咪Whiskers。
- 依此类推。

每一行包含表中所有列的值（尽管有时某个值可能被明确标记 (token)为未知或`NULL`）。单行中这些值的集合构成了关于一个特定实体的完整信息单元。

### 行作为记录

你会经常听到**记录**这个词与**行**同义使用。记录本质上就是单行中包含的信息。它是一组相关数据字段（列值），作为一个单元被处理。因此，`Pets`表中的第一行就是名为Fido的宠物的*记录*。

这是一个可视化表示：

> 一个简单的表格显示了列（标题）和行（记录）。高亮显示的第一行表示宠物“Fido”的完整记录。

这个道理很基本：列定义了*框架*，而行提供了*实际数据*，每行表示一个符合该框架的独立项或实体。正如我们很快会看到的那样，保证每行都有一个单一标识（比如这里使用`PetID`）是关系数据库设计的一个主要方面，这引出了主键的理念。眼下，请着重领会行是表格的一个横向切面，它保存了表格正在记录的某个特定事项的所有信息。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本广泛使用的数据库系统教材，涵盖关系模型中的关系、属性和元组。
