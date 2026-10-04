---
course: "introduction-to-databases"
chapter: "overview-nosql-databases"
lesson: "column-family-stores"
sourceId: 1616
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-4-overview-nosql-databases/column-family-stores"
title: "列族存储"
description: "了解列族数据库的概况，这类数据库将数据存储在列中而非行中。"
order: 5
plots: []
sourceHash: "b09c810b8f1f5fbfd62a20415b0273933dd04d9f356ac8bfcd063b23dd7762cf"
sourceCorrections: []
---

想象一下，您对表的传统观念被颠覆了。与其关注相关信息的行，不如关注列呢？这就是列族存储的基本理念，它是一种NoSQL数据库，用于应对特定类型的数据挑战，尤其是在大规模场景下。

这些数据库将数据组织成**列族**，您可以将其大致理解为行的容器。然而，与关系表不同，列族中的行不需要拥有相同的列集合。数据主要基于列存储，这使得某些操作效率很高。

### 列族存储的结构

让我们分解一下典型的组成部分：

1. **键空间（Keyspace）：** 这是最外层的容器，类似于关系模型中的模式或数据库。它将不同的列族组合在一起。
2. **列族（Column Family）：** 这是一组行的集合。列族中的每一行都由一个唯一的\*\*行键（Row Key）\*\*标识。可以将其视为一个表，但结构非常灵活。一个列族将相关列组合在一起。
3. **行键（Row Key）：** 这个唯一标识符的功能类似于关系表中的主键。它是定位列族中特定行的主要方式。
4. **列（Column）：** 这里开始变得有意思了。此模型中的列通常是一个元组，包含名称（或键）、值，通常还有时间戳。特定行键的数据以这些列的集合形式存储。重要的是，同一列族中的不同行可以包含完全不同的列。

以一个`UserProfile`列族为例。一个用户（由`RowKey: user123`标识）可能包含`email`、`last_login`和`city`等列。另一个用户（`RowKey: user456`）可能包含`email`、`last_login`和`preferred_language`等列。无需预先定义所有可能的列，也不会因为存储不适用于特定行的`null`值而浪费空间。

> `UserProfile`列族的一个简化视图。请注意，`user123`有`city`，`user456`有`preferred_language`，而`user789`有`status`，这体现了由唯一行键标识的行内部的可变结构。

### 为什么使用列族存储？

列族数据库的结构使其特别适合某些任务：

- **可扩展性：** 它们通常设计为在商品硬件的分布式集群上运行，使其能够水平扩展，以处理大量数据和高流量负载。
- **写入性能：** 许多列族数据库都针对高写入吞吐量 (throughput)进行优化，使其适合快速生成大量数据的应用，例如日志系统或传感器数据收集。
- **稀疏数据：** 它们能有效地处理那些单个记录可能拥有许多潜在属性，但对于任何给定记录只填充了少数属性的数据集（例如`UserProfile`示例）。对于不存在的属性，您无需支付存储开销。
- **面向列的查询：** 跨多行获取特定列的数据可以非常快速，因为相关列值可以连续存储在磁盘上。例如，从`UserProfile`列族获取所有电子邮件地址将是高效的。

### 与关系数据库的对比

在传统的关系数据库中，数据是逐行存储的。如果您想为所有用户只检索一列数据（例如，电子邮件地址），数据库通常必须读取每行的所有数据，包括您未请求的列，然后筛选出电子邮件地址。列族存储通过主要按列（在列族内部）组织数据，因此对于此类查询，通常可以直接访问所需的列数据。

### 常见示例

一些知名的列族数据库包括：

- **Apache Cassandra：** 因其高可用性和线性可扩展性而被广泛使用，常用于大规模网络应用、物联网数据存储和实时数据处理。
- **HBase：** 构建在Hadoop生态系统之上，HBase设计用于海量数据集（数十亿行，数百万列），并提供快速的随机读写访问。

当关系数据库的严格结构不适合您数据的规模或特性时，列族存储提供了一种有力的替代方案，尤其是在处理宽而稀疏的数据集或需要高写入性能和可扩展性时。它们在查询通常涉及从大量行中获取特定列子集的情况下表现出色。

## 参考资料

- [NoSQL Distilled: A Brief Guide to the Emerging World of Polyglot Persistence](https://martinfowler.com/books/nosql.html) — Pramod J. Sadalage and Martin Fowler (2012)
  Publisher: Addison-Wesley Professional
  为NoSQL数据库（包括列族存储）提供了清晰、高级的介绍，适合初学者。
- [Apache Cassandra Documentation](https://cassandra.apache.org/doc/) — The Apache Software Foundation (2024)
  官方文档，提供Apache Cassandra的全面信息，这是列族存储的一个重要实例。
- [Bigtable: A Distributed Storage System for Structured Data](https://www.usenix.org/conference/osdi-06/bigtable-distributed-storage-system-structured-data) — Fay Chang, Jeffrey Dean, Sanjay Ghemawat, Wilson C. Hsieh, Deborah A. Wallach, Mike Burrows, Tushar Chandra, Andrew Fikes, Robert E. Gruber (2006)
  Journal: 7th USENIX Symposium on Operating Systems Design and Implementation (OSDI '06); Publisher: USENIX Association; Pages: 205-218
  描述了Google的Bigtable，该系统启发了许多列族数据库，例如HBase。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  在分布式系统背景下，提供了各种数据库类型（包括列族模型）的详细信息。
