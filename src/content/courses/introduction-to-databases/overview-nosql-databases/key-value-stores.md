---
course: "introduction-to-databases"
chapter: "overview-nosql-databases"
lesson: "key-value-stores"
sourceId: 1614
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-4-overview-nosql-databases/key-value-stores"
title: "键值存储"
description: "了解简单的键值存储，它们针对检索与唯一键关联的值进行了优化。"
order: 4
plots: []
sourceHash: "8426f6982f38f69a2a811c0abc75f4f1b2ed342ca4354acc58df330301c4d28a"
sourceCorrections: []
---

设想最简单的数据存储和检索方式：你有一个唯一的标识符（“键”）和与其相关联的数据（“值”）。这就是键值存储的核心，它是NoSQL数据库的基本类型之一。可以把它想象成一个庞大而高效的字典或哈希映射，你通过查找一个词（键）来找到它的定义（值）。

### 主要思想：简单与快速

键值存储围绕一项操作构建：检索与给定键关联的值。这种直接查找机制使它们在特定任务中非常快速。与关系型数据库不同，在关系型数据库中可能需要根据特定条件搜索行和列，键值存储通常允许你在知道键的情况下，一步到位地获取数据。

结构简单明了：

1. **键：** 唯一的字符串、整数或其他标识符，用于定位数据。唯一性是必需的；每个键都准确地映射到集合中的一个值。
2. **值：** 实际存储的数据。键值存储的一个重要特点是其在值方面的灵活性。数据库通常将值视为一个不透明的数据块。它可以是任何形式：
   - 简单的字符串（如用户的姓名）
   - 整数（如计数器）
   - JSON对象（包含结构化的用户资料信息）
   - 序列化对象
   - 图像或其他二进制数据

数据库本身通常不强制要求值 *内部* 的任何结构。应用程序需要知道该值代表什么以及如何解析它。

> 键值存储的简化视图。应用程序使用唯一键来存储和检索关联的值，这些值可以是简单的文本、JSON等结构化数据、数字，甚至二进制数据。

### 键值存储的适用场景？

它们的简单与快速使它们非常适合特定情况：

- **缓存：** 这是一个非常普遍的用途。来自较慢数据库（如关系型数据库）的频繁访问数据或计算开销大的操作，可以存储在快速的键值存储中。在查询主数据库之前，应用程序会使用特定键（例如 `product:details:567`）检查缓存。如果数据存在，它会快速返回，从而避免较慢的查找。示例包括Redis和Memcached。
- **会话管理：** Web应用程序需要在多个请求之间跟踪用户会话。用户的会话ID可以作为标识符，会话数据（登录状态、用户偏好、购物车内容）可以作为值存储。这使得用户每次发起新请求时，都能快速检索会话信息。
- **用户资料：** 需要快速访问的基本用户资料信息（如用户设置或偏好）可以以用户ID作为键存储。
- **实时数据：** 游戏中的排行榜或简单的计数器（如帖子点赞数）可以在键值存储中有效管理，键值存储通常提供原子操作来增加数字。

### 优势与考虑

**优势：**

- **高性能：** 针对基于键的简单GET/PUT/DELETE操作进行了优化。
- **可伸缩性：** 通常比关系型数据库更容易进行水平伸缩（添加更多服务器），因为数据通常可以根据键分布在不同节点上。
- **灵活性：** 值无模式的特性允许存储不同类型的数据，而无需预先定义结构。

**需要考虑的事项：**

- **查询局限性：** 基于 *值* 内容的查询通常效率低下或不直接支持。通常需要知道键才能有效地检索数据。涉及多个键或关系的复杂查询通常不是简单键值存储的强项。
- **数据关系：** 它们不像关系型数据库那样，通过外键内在管理不同数据之间的关系。关系管理通常由应用程序逻辑负责。
- **一致性模型：** 分布式键值存储可能提供不同级别的数据一致性（例如，最终一致性），应用程序需要对此加以考虑。

键值存储的常见示例包括 **Redis**、**Memcached** 和 **Amazon DynamoDB** （它也具备文档数据库功能）。

总之，当主要需求是使用唯一标识符存储和检索数据时，键值存储提供一种简单、高性能的解决方案。它们在缓存和会话管理等场景中表现出色，在这些场景中，速度和简单查找非常重要。它们与结构化表格有很大差异，优先考虑特定访问模式的速度和可伸缩性。

## 参考资料

- [NoSQL Distilled: A Brief Guide to the Emerging World of Polyglot Persistence](https://martinfowler.com/books/nosql.html) — Pramod J. Sadalage, Martin Fowler (2012)
  Publisher: Addison-Wesley Professional
  简明介绍各种NoSQL数据库类型，包含对键值存储的专门讲解。
- [Redis Documentation](https://redis.io/docs/) — Redis Labs (2024)
  Redis官方指南，通过缓存和会话管理示例展示其作为高性能键值存储的用途。
- [Amazon DynamoDB Developer Guide](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  官方指南，详细介绍DynamoDB作为可伸缩键值和文档数据库服务的功能。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  全面检查数据系统架构，包含与键值存储、分布式系统和一致性模型设计和实现原则相关的章节。
