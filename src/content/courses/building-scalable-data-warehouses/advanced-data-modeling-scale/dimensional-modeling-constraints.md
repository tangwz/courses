---
course: "building-scalable-data-warehouses"
chapter: "advanced-data-modeling-scale"
lesson: "dimensional-modeling-constraints"
sourceId: 8026
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale/dimensional-modeling-constraints"
title: "大数据中的维度建模限制"
description: "评估传统星型模式在PB级环境中的局限性"
order: 1
plots: []
sourceHash: "6e8df2032f5c697b75bf72fa810a42a3e4d9a1d9565f0a0fc380d4f44f8eb980"
sourceCorrections: []
---

传统维度建模，特别是拉尔夫·金博尔推广的星型模式，依赖于一个基本前提：维度表远小于事实表。这一前提便于数据库引擎将维度载入内存，并对事实表执行哈希连接，同时保持极低的I/O延迟，特别是在单个服务器上的SQL Server或Oracle等对称多处理（SMP）环境中。

然而，在Snowflake、Redshift或BigQuery等大规模并行处理（MPP）架构中，这一前提常常不成立。当数据集达到PB级时，维度表往往会增长到数亿甚至数十亿行。一个全球性B2C平台的客户维度或物联网设备维度，可能轻易超出单个计算节点的内存容量。此时，星型模式的结构刚性会带来严重的性能瓶颈，这些瓶颈与网络拓扑和存储不可变性有关。

### 分布式连接与数据重排成本

大数据中维度建模的主要限制是分布式连接的成本。在无共享架构中，数据物理上分散在多个计算节点（或分片）上。要将事实表与维度表连接，查询优化器必须确保匹配的键位于同一节点上。

如果维度表较小（通常在10-50GB以下，具体取决于平台），引擎会使用**广播连接**。整个维度表会被复制到每个计算节点。这效率高，因为大型事实表碎片保持在本地，并且连接并行发生，无需移动事实数据。

当维度表大到无法广播时，引擎必须执行**数据重排连接**。系统根据连接键的哈希值，通过网络重新分配事实数据和维度数据。此操作会占满网络带宽，这是分布式系统中最稀缺的资源。

查询成本$C$大致遵循此函数，其中网络传输$N$通常远超CPU $P$和磁盘I/O $D$：

$C_{查询} \approx \sum (D_{扫描} + P_{执行} + N_{重排})$

在一个具有多个大型维度（例如高基数的用户ID、设备ID和会话ID）的高度规范化星型模式中，单个查询可能触发多次数据重排阶段，在集群中层叠传输数据，并产生拖慢者（等待数据的节点）。

> 数据重排连接的运作方式：分区键不匹配迫使数据在网络层传输，引入本地连接中不存在的延迟。

### 不可变性与更新并发问题

现代云数据仓库使用由对象存储（S3、GCS、Azure Blob）支持的列式存储格式。这些底层文件是不可变的。要更新一条记录，系统不会修改某个特定块；它会将旧的微分区标记 (token)为过时，并写入一个新的微分区，其中包含更新后的数据和现有未更改的数据。

这种架构与慢变维度（SCD）类型1（覆盖）和类型2（添加行）所需的频繁更新模式相冲突。

在传统的星型模式中，`客户`维度可能带有`上次登录日期`或`当前会话状态`等易变属性。对一个拥有数十亿行的维度中的这些属性进行频繁更新，会触发一种被称为**写入放大**的现象。

1. **聚类中断：** 重新写入微分区会降低表的聚类比率，需要耗费资源的后台重新聚类操作（例如Snowflake的自动聚类或BigQuery的剪枝效果）。
2. **锁定与排队：** 尽管许多现代平台使用多版本并发控制（MVCC），但对单个大型维度表的频繁更新仍可能导致事务序列化，阻止同时进行的ETL作业修改该维度。

与面向行的数据库中`UPDATE`只是局部指针更改不同，在列式MPP系统中，在星型模式中管理高速变更会产生显著的计算开销。

### 模式刚性与演变

第三个主要限制是业务逻辑与物理表结构的耦合。在星型模式中，业务规则融入了事实表和维度表的粒度。如果源系统发生更改，例如上游CRM将“地址”拆分为“账单地址”和“配送地址”，下游维度表就需要执行`ALTER TABLE`命令。

在大数据环境中，对宽表进行模式演变操作并非仅限于元数据的操作。根据平台和具体更改（例如更改数据类型），这可能强制进行全表重写。对于PB级仓库，全表重写计算成本高昂，可能耗费数小时或数天，实际上导致该数据资产的停机时间。

此外，星型模式强制实体之间建立紧密关系。这种刚性在处理半结构化数据或稀疏数据集时会失效，这些数据常见于点击流或遥测日志中。将复杂的JSON层次结构强制扁平化到维度表中会导致稀疏列（数千个NULL值），虽然可以有效压缩，但会使查询逻辑和ETL维护更加复杂。

### 宽表中的元数据延迟

大数据建模常会产生“蜈蚣式”事实表（即包含数百个外键，连接到数百个维度的表），或拥有数百个列的“怪兽式”维度。

查询优化器依赖元数据（最小值/最大值、NULL计数、不同值计数）来剪枝分区。当维度表变得过宽（例如500+列）时，元数据开销会增加。优化器需要更多时间来编译查询计划，分析哪些分区可以被忽略。

尽管列式存储允许只选择特定列，但初始元数据扫描本质上是面向行的（扫描微分区的头部信息）。维度模型中的过度宽度可能导致编译时间超过亚秒级查询的执行时间。

### 重新评估规范化

这些限制表明，金博尔提倡的严密反规范化并非MPP系统的默认最佳实践。在某些场景下，对大型维度进行规范化（雪花化）是更优选择，以减小被广播表的大小。在其他高吞吐量 (throughput)场景中，将易变属性从主要维度移至快速变化的“热”表（创建混合模式）可避免写入放大。

这种转变是从最小化存储（3NF）或最大化可读性（星型模式）转向**最小化数据传输**（分布设计）和**最小化文件更迭**（分区策略）。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling, 3rd Edition](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  关于维度建模、星型模式和数据仓库设计原则的指南。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  关于分布式系统、无共享架构、数据分区和分布式连接策略的参考资料。
- [Join Performance Considerations](https://docs.snowflake.com/en/user-guide/query-optimization) — Snowflake (2024)
  Publisher: Snowflake
  Snowflake关于查询优化的官方文档，涵盖MPP环境中的广播和混洗等连接策略。
