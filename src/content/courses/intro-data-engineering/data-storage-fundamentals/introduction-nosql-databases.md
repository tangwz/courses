---
course: "intro-data-engineering"
chapter: "data-storage-fundamentals"
lesson: "introduction-nosql-databases"
sourceId: 5557
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-4-data-storage-fundamentals/introduction-nosql-databases"
title: "NoSQL 数据库简介"
description: "了解不同类型的 NoSQL 数据库及其特点。"
order: 3
plots: []
sourceHash: "45b634885ef04512463de93e618c82a66f65c1e4197a2949373ddeb1369a899b"
sourceCorrections: []
---

关系型数据库提供了一种强大且广为人知的方式来以表格形式存储结构化数据，但它们并非总是适合所有情况，尤其是在处理海量数据集、多样数据形式以及需要大规模高速性能时。这时，NoSQL 数据库便应运而生。

“NoSQL”一词常被理解为“不只是 SQL”。它不一定意味着完全放弃 SQL，而是采用更广泛的数据库技术，这些技术旨在解决关系型数据库可能面临的难题。可以将它们看作是为特定数据存储和检索问题而设计的专用工具。

### 为什么选择 NoSQL？主要原因

NoSQL 数据库的兴起，主要受大规模 Web 应用和大数据处理的需求推动：

1. **可扩展性：** 许多 NoSQL 数据库从一开始就设计为可水平扩展。这意味着你可以通过向集群添加更多服务器（通常是更便宜的通用硬件）来提升容量，而不是升级单个、大型、昂贵的服务器（垂直扩展），这对于许多关系型数据库来说是传统方法。
2. **灵活性（模式设计）：** 关系型数据库强制使用严格的模式。你需要预先定义表和列，所有数据都必须符合该结构。NoSQL 数据库通常提供动态或灵活的模式。这在处理无法整齐地放入表格、快速变化或来自不同来源（如用户生成内容、传感器数据或日志）的数据时很有利。你可以在不为每种可能性预定义列的情况下，存储不同项目的不同属性。
3. **处理多样数据类型：** 它们擅长存储非结构化（如文本文档、图像）或半结构化数据（如 JSON 或 XML），以及结构化数据。这些数据纯粹在关系型表格中管理起来可能会很笨拙。
4. **针对特定工作负载的性能：** 某些 NoSQL 数据库针对特定访问模式进行了优化，例如极快的键查找（键值存储）或遍历复杂关系（图数据库），在这些任务上可能胜过通用关系型数据库。

### NoSQL 的主要理念

与传统关系型数据库（如 PostgreSQL 或 MySQL）相比，NoSQL 系统通常在几个重要方面有所不同：

- **数据模型：** 它们不只使用表格，而是采用键值对、文档、列族或图等多种模型。
- **模式：** 通常无模式或模式灵活，允许数据结构随时间更轻松地改变。
- **扩展方式：** 主要依赖于跨多台服务器的水平扩展。
- **一致性：** 关系型数据库通常优先考虑强一致性（ACID 特性 - 原子性、一致性、隔离性、持久性），而许多 NoSQL 数据库提供可调整的一致性级别，有时选择“最终一致性”以获得更高的可用性和分区容错性，尤其是在分布式系统中。最终一致性意味着，如果没有新的更新，数据的所有副本最终将收敛到相同的值。

### NoSQL 数据库的类型

NoSQL 并非单一产品，而是一个涵盖几种不同数据库类型的类别。以下是主要的几个类别：

#### 1. 键值存储

这是最简单的 NoSQL 数据库形式。数据以键值对集合的形式存储，很像编程中的字典或哈希映射。你提供一个唯一的键，数据库会返回关联的值（可以是简单的字符串、数字或复杂对象）。

- **工作原理：** 可以理解为 `get(key)` 和 `put(key, value)`。
- **优势：** 基于键的简单查找、写入和删除速度极快。高度可扩展。
- **使用场景：** 缓存网络会话、用户偏好、实时排行榜。
- **例子：** Redis、Memcached、Amazon DynamoDB（也具有文档特性）。

#### 2. 文档数据库

文档数据库以文档格式存储数据，常见的有 JSON (JavaScript Object Notation)、BSON (Binary JSON) 或 XML。每个文档都是自包含的，可以具有复杂、嵌套的结构。文档通常分组到集合中（类似于表格）。

- **工作原理：** 存储和检索整个文档。文档可以在同一集合中具有不同结构。通常允许基于文档内字段进行查询。
- **优势：** 模式灵活，与面向对象编程结构自然映射，适用于不断演进的应用。
- **使用场景：** 内容管理系统、用户资料、产品目录、移动应用数据。
- **例子：** MongoDB、Couchbase、ArangoDB。

以下是 `users` 集合中两个文档的简单例子：

```json
{
  "_id": "user123",
  "name": "Alice",
  "email": "alice@example.com",
  "interests": ["data engineering", "python"]
}

{
  "_id": "user456",
  "name": "Bob",
  "city": "New York",
  "last_login": "2023-10-27T10:00:00Z"
}
```

> JSON 格式的用户文档示例。请注意 Bob 有 `city` 和 `last_login`，而 Alice 有 `email` 和 `interests`。

#### 3. 列族存储（宽列存储）

这些数据库以行和列的形式存储数据，但它们针对按列而非按行访问数据进行了优化。给定行的数据可能分布在多个节点上，并按列族（列的逻辑分组）进行分组。

- **工作原理：** 高效地读取和写入许多行的特定列。可以处理具有大量列的行，其中给定行中的许多列可能为空。
- **优势：** 对于写入密集型工作负载以及访问许多行中列子集的查询具有高度可扩展性。适用于非常大的数据集。
- **使用场景：** 分析、时间序列数据（如日志或指标）、推荐引擎、具有稀疏列的大型数据集。
- **例子：** Apache Cassandra、Google Bigtable、Apache HBase。

#### 4. 图数据库

图数据库专门用于存储和遍历关系。数据被建模为节点（实体）、边（连接节点的联系）和属性（节点和边的特征）。

- **工作原理：** 侧重于数据点之间的连接。查询通常涉及遍历这些连接。
- **优势：** 高效处理高度互连的数据和复杂的关系查询。
- **使用场景：** 社交网络、欺诈检测、推荐系统（例如，“购买此商品的顾客也购买了…”）、知识图谱、网络图。
- **例子：** Neo4j、Amazon Neptune、ArangoDB（多模型）。

> 显示用户、产品及其关系（朋友、购买、浏览）的简单图示，类似于在图数据库中的存储方式。

### 选择合适的工具

NoSQL 数据库提供了强大的替代方案，但它们并非关系型数据库的普遍替代品。选择哪种数据库很大程度上取决于你的具体需求：

- **数据结构：** 你的数据是高度结构化且呈表格形式（倾向于 SQL），还是半结构化、非结构化或图状（倾向于 NoSQL）？
- **可扩展性需求：** 你是否预计需要大规模扩展，并需要轻松进行水平扩展（倾向于 NoSQL）？
- **模式灵活性：** 你的数据模式是否经常变化（倾向于 NoSQL）？
- **查询模式：** 你的查询是简单的键查找（键值存储）、复杂的关系遍历（图数据库），还是对结构化数据进行标准分析查询（SQL）？
- **一致性要求：** 你是否需要每笔事务都提供严格的 ACID 保证（倾向于 SQL），还是可以容忍最终一致性以获得更高的可用性（通常是 NoSQL 的一种权衡）？

通常，组织会混合使用 SQL 和 NoSQL 数据库（一种多语言持久化方法），为应用程序或数据工作负载的每个具体部分选择最佳技术。理解每种类型的优缺点有助于你针对数据存储位置做出明智的决定。

## 参考资料

- [NoSQL Distilled: A Brief Guide to the Emerging World of Polyglot Persistence](https://martinfowler.com/books/nosql.html) — Pramod J. Sadalage and Martin Fowler (2012)
  Publisher: Addison-Wesley
  对 NoSQL 概念、不同数据库类型以及多模型持久化方法的简洁而权威的介绍，适合初学者。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.barnesandnoble.com/w/designing-data-intensive-applications-martin-kleppmann/1125292419) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media, Incorporated
  深入探讨了各种数据存储和处理系统，包括对不同 NoSQL 数据模型、一致性和可伸缩性的详细论述，是数据工程的基础。
- [Dynamo: Amazon's Highly Available Key-Value Store](https://dl.acm.org/doi/10.1145/1294261.1294281) — Giuseppe DeCandia, Deniz Hastorun, Michael Jampani, Gunavardhan Kakulapati, Avinash Lakshman, Alex Pilchin, Swaminathan Sivasubramanian, Peter Vosshall, and Werner Vogels (2007)
  Journal: SOSP '07: Proceedings of the ACM Symposium on Operating Systems Principles; Publisher: ACM; Pages: 205-220; DOI: [10.1145/1294261.1294281](https://doi.org/10.1145/1294261.1294281)
  描述亚马逊分布式键值存储的开创性论文，引入了最终一致性和冲突解决等概念，对于理解分布式 NoSQL 系统至关重要。
