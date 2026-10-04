---
course: "introduction-to-databases"
chapter: "understanding-databases-storage"
lesson: "database-management-systems-dbms"
sourceId: 1553
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-1-understanding-databases-storage/database-management-systems-dbms"
title: "数据库管理系统（DBMS）"
description: "了解用于创建、管理和与数据库交互的软件（DBMS）。"
order: 4
plots: []
sourceHash: "27be3cded22d1ccb44ea659f829d9bff8106de1ab5d7c588251fbbe9eaafa9a1"
sourceCorrections: []
---

将数据存储在散乱的文件中常常会导致信息重复、查找困难以及难以保持一致性等问题。想象一下管理一个庞大的图书馆，每本书都随意堆在地板上——要找到某个特定的句子几乎是不可能的！数据库提供了有序的结构，就像书架和目录一样，但仍然需要一个系统来管理图书馆本身。这就是数据库管理系统（简称DBMS）的作用所在。

可以把DBMS看作是集高效图书管理员和整个图书馆系统于一体的软件。它不是数据库本身（数据的集合），而是一个复杂的软件应用，允许你、你的应用和其他用户创建、访问、管理和保护数据库中存储的数据。没有DBMS，数据库只是一组静态数据；DBMS使其变得活跃。

### DBMS实际做什么？

DBMS处理多项基本任务，使数据库有用且可靠：

1. **定义数据结构：** 在存储数据之前，你需要定义数据将如何存储。DBMS提供工具（通常使用一组名为数据定义语言或DDL的专用命令）来指定数据库的结构。这包括定义表、表中包含的列、每列保存的数据类型（如数字、文本或日期）以及不同数据之间的关系。这就像给图书管理员一份蓝图，说明图书馆的书架和分区应如何组织。
2. **数据操作：** 结构搭建好后，主要工作就是与数据交互。DBMS处理以下请求：

   - **添加（插入）新数据：** 把新书放到正确的书架上。
   - **检索（查询）现有数据：** 根据查询找到特定书籍或信息。
   - **修改（更新）现有数据：** 更正图书记录中的拼写错误。
   - **移除（删除）数据：** 将一本书永久性地从流通中移除。
     这些操作通常使用数据操作语言（DML）执行。最常用的DML是SQL（结构化查询语言），稍后你将学到更多关于它的知识。
3. **访问与安全管理：** 并非每个人都应该能走进图书馆，在书上乱涂乱画，或者不办理借阅手续就带回家。DBMS控制谁可以访问数据库以及他们被允许做什么（只读、更新特定数据等）。它执行安全规则，保护数据免受未经授权的访问或修改。
4. **确保数据完整性与一致性：** DBMS执行规则，确保数据保持准确和一致。例如，它可以确保订单记录始终引用有效的客户记录，或者产品库存水平不会降到零以下。这些规则有助于防止简单文件系统中常见的错误。
5. **处理并发访问：** 通常，多个用户或应用需要同时访问数据库。想象一下几个图书管理员同时尝试更新同一张目录卡！DBMS管理这种并发，确保同时操作不会相互干扰并损坏数据。
6. **备份与恢复：** 如果图书馆发生水灾怎么办？一个好的系统需要备份方案。DBMS提供定期备份数据库并在硬件故障、软件崩溃或其他灾难时恢复数据库的机制。

### DBMS层次结构

你可以将DBMS视为用户（或应用）与存储在磁盘上的实际数据之间的中间层。

> 此图说明了用户和应用如何与DBMS交互，DBMS随后处理从物理存储文件读取和写入的细节。

### DBMS软件示例

你会遇到许多不同的DBMS产品。一些常见的例子包括：

- **关系型DBMS（通常使用SQL）：** MySQL、PostgreSQL、Microsoft SQL Server、Oracle Database、SQLite（一种轻量级版本，常用于网络浏览器和手机等应用中）。
- **NoSQL DBMS（不同数据模型）：** MongoDB（文档型）、Redis（键值型）、Cassandra（列族型）。你将在本课程的后面学到更多关于这些内容。

记住这个区别很重要：MySQL是*DBMS软件*；你使用MySQL创建的实际表和数据集合是你的*数据库*。

实质上，DBMS是驱动现代数据管理的引擎。它将有序数据库的思路付诸实践，并提供必要的工具和保障，使其成为一种实用、安全、高效的信息处理方式，远超基本文件存储的能力。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本广泛使用的教材，全面涵盖了数据库管理系统，包括数据模型、设计、查询语言和事务管理，直接支持了所讨论的核心概念。
- [NoSQL Distilled: A Brief Guide to the Emerging World of Polyglot Persistence](https://www.martinfowler.com/books/nosql.html) — Pramod J. Sadalage, Martin Fowler (2012)
  Publisher: Addison-Wesley Professional
  本书简明概述了NoSQL数据库，解释了其各种类别和应用场景，是扩展关系型系统之外的DBMS类型理解的良好资源。
