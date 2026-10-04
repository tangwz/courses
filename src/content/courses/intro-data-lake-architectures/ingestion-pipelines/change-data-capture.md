---
course: "intro-data-lake-architectures"
chapter: "ingestion-pipelines"
lesson: "change-data-capture"
sourceId: 8006
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-3-ingestion-pipelines/change-data-capture"
title: "变更数据捕获 (CDC)"
description: "通过日志捕获数据库变更，以在数据湖中保持同步状态。"
order: 2
plots: []
sourceHash: "90765ff55da1c7565a21322d49056f500dabbc2cc09305122b47e437ff733f1f"
sourceCorrections: []
---

尽管批处理数据摄取为移动历史快照提供了一种可靠方法，但通常无法满足现代分析应用程序的延迟要求。当业务需要在几分钟而非几天内响应库存变动、欺诈预警或客户互动时，等待夜间的批量加载是不足的。此外，随着数据量的增长，重复的全表快照变得非常昂贵。为了解决这些限制，数据工程师使用变更数据捕获 (CDC)。

CDC 是一种设计模式，用于识别和跟踪源系统中的数据变更，以便将这些变更应用于下游存储库。在数据湖的背景下，CDC 将数据库集成策略从“复制状态”转变为“流式传输事件”。

### 变更检测方法

实现 CDC 主要有两种机制：基于查询的轮询和基于日志的提取。了解其中的区别对设计可扩展的数据管道大有裨益。

**基于查询的 CDC** 依赖于应用层。它要求源表具有一个跟踪上次修改时间（例如 `updated_at`）或自增 ID 的列。数据摄取管道定期运行 SQL 查询，以获取跟踪列值大于上次运行高水位标记 (token)的记录。

$\text{SELECT } * \text{ FROM } \text{orders} \text{ WHERE } \text{updated\_at} > \text{'2023-10-27 10:00:00'}$

这种方法易于实现，但存在明显的缺点：

1. **漏掉删除：** 如果源数据中的记录被硬删除，SQL 查询将不会返回该记录。目标数据湖将无限期地保留过时的记录。
2. **性能影响：** 频繁轮询会增加源数据库查询引擎的负担。
3. **数据一致性：** 它无法捕获中间状态。如果一条记录在轮询间隔期间更新了三次，管道只会捕获最终状态，而丢失了转换历史。

**基于日志的 CDC** 被广泛认为是生产数据湖的标准方式，它直接与数据库事务日志交互（例如 PostgreSQL 中的预写日志 WAL、MySQL 中的二进制日志 Binlog 或 Oracle 中的重做日志 Redo Log）。每个数据库在确认写入之前都会将事务提交到日志文件。基于日志的 CDC 工具充当读取此日志流的客户端。

此方法会按发生的精确顺序捕获每个事件（插入、更新、删除）。它对源数据库的负载很小，因为它读取的是文件系统而不是执行 SQL 查询。

> 基于日志的架构将提取过程与数据库查询引擎分离，从而实现实时事件流传输，而不会降低性能。

### CDC 事件的构成

在基于日志的管道中，流经系统的数据不再是简单的行；它是一个包含数据和操作元数据的“信封”。标准的 CDC 消息通常包含：

- **操作类型：** 指示变更操作是创建 (c)、更新 (u) 还是删除 (d)。
- **时间戳：** 事务发生的时间。
- **前镜像：** 变更发生前行的状态（对更新很有用处）。
- **后镜像：** 变更发生后行的状态。

这种结构允许数据湖在任何时间点重建数据库状态。在 Medallion 架构中，这些原始 CDC 事件直接落入**青铜层**。我们不会立即尝试合并或去重这些事件。青铜层充当从源接收到的每个变更的不可变历史日志。

### 将变更传播到白银层

一旦原始事件安全地存储在青铜层（通常以 JSON 或 Avro 文件形式存储），工程上的难题就转向将这些变更应用于**白银层**表。在这里，Apache Iceberg 或 Delta Lake 等开放表格式变得必不可少。

在标准文件系统（如 S3 上的原始 Parquet）中，无法更新特定行。您将不得不重写整个文件。表格式使得数据湖上的 ACID 事务成为可能，允许进行独特的 `MERGE` 操作。

将 CDC 数据源处理到白银表的标准模式涉及以下逻辑：

1. **去重：** 在分布式系统中，事件可能乱序到达或重复。我们必须按事务时间戳对事件进行排序，并在当前微批次中仅保留特定主键的最新变更。
2. **合并操作：** 我们使用 `MERGE INTO` SQL 命令。此命令根据主键将传入的事件批次与目标表连接起来。
   - **当匹配且操作为删除时：** 从目标表中删除该行。
   - **当匹配且操作为更新时：** 更新目标列。
   - **当不匹配且操作为插入时：** 插入新行。

此过程有效地将数据湖与源数据库同步。

### 处理硬删除

处理删除是基于日志的 CDC 相对于批量快照最主要的优势。当源中删除一行时，日志会发出一个包含被删除记录主键的删除事件。

在数据湖中，处理此事件有两种策略：

1. **硬删除：** 从白银表中物理删除该行。这可以保持与源的精确镜像，但会阻止对已删除项目的历史分析。
2. **软删除：** 在白银表中添加一个元数据列 `is_deleted`。管道不删除记录，而是将此标志更新为 `true`。

软删除在数据工程中通常更受欢迎，因为它们保留了历史记录。分析师可以过滤掉已删除的记录以进行当前状态报告 ($\text{其中 is\_deleted} = \text{假}$)，同时保留审计记录何时以及为何被删除的能力。

### 重新快照的挑战

尽管 CDC 能有效地捕获增量变更，但纯粹基于日志的管道通常是空开始的。要初始化数据湖，需要进行“历史加载”或“快照”。

一种生产模式涉及混合方法：

1. **快照阶段：** 执行源表的整体一次性批量导出。
2. **日志阶段：** CDC 连接器被配置为从快照完成时刻对应的特定位置（日志序列号或偏移量）开始读取日志。

这确保了零数据丢失，并防止了历史数据转储与正在进行的流之间出现空隙。像 Debezium 这样的工具会自动处理这种协调，无需手动干预即可从快照模式切换到流模式。

批处理和 CDC 之间的选择最终视数据的波动性和消费者对数据新鲜度的要求而定。然而，随着数据湖越来越多地作为运营机器学习 (machine learning)模型和近实时仪表板的后端，向基于日志的 CDC 的架构转变正在成为标准实现方式。

## 参考资料

- [Debezium Documentation](https://debezium.io/documentation/) — The Debezium Community (N/A)
  Debezium的官方文档，一个开源的变更数据捕获平台。它涵盖了基于日志的CDC连接器的设置、配置和操作方面，包括它们如何处理初始快照和从各种数据库进行的持续事件流。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书深入探讨了可靠、可扩展和可维护数据系统背后的原理。关于复制、事务日志和流处理的章节为基于日志的变更数据捕获机制提供了基础理解。
- [Delta Lake Documentation](https://docs.delta.io/latest/index.html) — The Delta Lake Community (N/A)
  Delta Lake的官方文档，一个为Apache Spark和大数据工作负载带来ACID事务的开源存储层。它对于理解如何在数据湖表上执行MERGE INTO操作以应用CDC事件很有用。
- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/) — The Apache Iceberg Community (N/A)
  Publisher: Apache Software Foundation
  Apache Iceberg的官方文档，一个用于大型分析数据集的开放表格式。它详细介绍了Iceberg如何支持模式演进、隐藏分区和表修改（包括MERGE INTO），这些对于在数据湖中管理CDC驱动的更新非常重要。
