---
course: "building-scalable-data-warehouses"
chapter: "mpp-architecture-storage-internals"
lesson: "micro-partitioning-metadata"
sourceId: 8024
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-1-mpp-architecture-storage-internals/micro-partitioning-metadata"
title: "微分区与元数据管理"
description: "自动微分区和元数据服务如何实现高效数据获取。"
order: 4
plots: ["plots/8024-0.json"]
sourceHash: "72cbeec8c11958859012c779a005a40bd9f425eabae0e9e6188e63c9e348b322"
sourceCorrections: []
---

传统数据仓库常依赖于工程师明确定义的静态分区方案。这通常包括按日期或区域划分大型事实表，以减少查询时扫描的数据量。这种方法虽然有效，但带来了局限性。如果查询模式改变，需要按不同维度筛选，静态分区结构就会失效。现代MPP平台，如Snowflake和BigQuery，通过自动微分区和对元数据服务的很大程度的依赖来处理这个问题。

## 微分区的架构

在现代云数据仓库中，数据不是以大型、整体的文件形式存储的。相反，系统自动将加载的数据划分为连续的存储单元，称为微分区。与可能表示文件目录的传统分区不同，微分区是存储在底层块存储（如AWS S3、Google Cloud Storage或Azure Blob Storage）中的独立文件对象。

每个微分区通常包含50 MB到500 MB的未压缩数据。这个大小最适合网络吞吐量 (throughput)，并让查询引擎只下载表的特定部分，而不是读取整个数据集。

微分区的重要特点包括：

- **不可变性：** 一旦写入，微分区就永不更改。看起来是更新数据的操作，实际上是创建包含更新行的新微分区，并将旧版本标记 (token)为过时。这种写时复制机制使得时间旅行和零拷贝克隆等功能得以实现。
- **列式组织：** 在微分区内部，数据按列存储。这使得引擎能够只反序列化`SELECT`子句所需的列。
- **自动压缩：** 系统分析微分区内每列的数据类型和基数，以便独立应用最有效的压缩算法。

## 元数据剪枝与区域图

MPP系统的性能表现很大程度上受其忽略数据能力的影响。这个过程被称为分区剪枝。系统实现这一点不是通过查看数据文件，而是通过查询独立的元数据层。

当数据写入微分区时，系统自动收集每列的统计信息，并将其存储到全局元数据服务中。常见指标包括：

- 最小值和最大值
- 不同值的数量
- NULL值的数量

当查询到达时，优化器会检查`WHERE`子句。它将查询谓词与元数据摘要进行比较。如果谓词中的值范围与微分区的最小/最大范围没有交集，则该分区会被完全跳过。

考虑一个基于时间戳的筛选查询：

$\text{SELECT count(*)} \text{ FROM events WHERE event\_ts} > \text{'2023-10-01'}$

优化器扫描元数据。如果某个微分区的最大`event_ts`是`'2023-09-30'`，引擎就能确定该文件中没有匹配的行。它会生成一个文件扫描列表，将该分区排除在外。

这个过程显著减少了I/O操作。剪枝效率可以用所选分区 $P_{selected}$ 与总分区 $P_{total}$ 的比率来表示：

$E_{pruning} = 1 - \frac{P_{selected}}{P_{total}}$

随着 $E_{pruning}$ 趋近于1，查询会变得更快、成本更低。

> 元数据服务扮演守门员的角色。它根据统计信息头部评估查询谓词，以决定需要获取哪些存储对象，防止不必要的I/O操作。

## 聚类与数据分布

元数据剪枝仅在数据良好排序时才有效。如果表是随机排序的，每个微分区的最小/最大范围很可能覆盖整个数据范围。在这种情况下，元数据检查会对每个分区都返回肯定结果，从而强制进行全表扫描。

**聚类**是指基于一个或多个列（聚类键）对数据进行物理排序。

1. **自然聚类：** 数据通常按插入时间自然聚类。如果您按顺序加载日志数据，`timestamp`列将自然排序。按时间筛选的查询将能高效剪枝。
2. **手动聚类：** 对于按非顺序高基数列（例如`user_id`或`transaction_id`）筛选的查询，可能需要明确地重新排序数据。

当数据有效聚类时，微分区的值范围是独立且窄的。当数据未聚类时，这些范围会显著重叠。

下面的图表展示了排序对分区重叠的影响。在未聚类情况下，查询值“50”可能会匹配几乎所有分区的范围。在聚类情况下，它会针对一个特定的子集。



![聚类对分区值范围的影响](plots/8024-0.json)



> 值重叠的比较。未聚类分区（红色）具有覆盖整个范围的宽泛的最小-最大值区间。已聚类分区（青色）具有狭窄、顺序的范围，这让查询引擎可以从扫描计划中排除文件。

## 计算聚类深度

为了评估表布局的状况，平台提供系统函数来计算聚类深度。深度代表了特定值平均重叠的微分区数量。

如果查询筛选`WHERE ID = 500`且聚类深度为1，系统将精确读取一个分区。如果深度为1000，它将读取1000个分区来找到相同的ID。

保持完美聚类成本很高。每次重新聚类表时，系统都必须重写微分区（请记住，它们是不可变的）。这会产生计算成本。您必须权衡后台重新聚类的成本与查询期间获得的计算节省。

## 模式演变的影响

由于微分区是带有各自头部和压缩字典的独立单元，现代MPP系统可以比传统系统更灵活地处理模式演变。

当您向表中添加列时，系统无需重写现有历史数据。它只需更新元数据来注册新列。旧的微分区保持不变。当查询从旧分区请求新列时，引擎会在运行时注入`NULL`值。模式更改后创建的新微分区将包含该列的物理数据。

这种逻辑模式与物理存储的分离使得非阻塞DDL操作成为可能，这是数据管道中持续集成的一个重要功能。

## 参考资料

- [The Snowflake Elastic Data Warehouse](https://dl.acm.org/doi/10.1145/2882903.2903746) — Benoit Dageville, Thierry Cruanes, Marcin Zukowski, Vadim Antonov, Artin Avanes, Jon Bock, Jonathan Claybaugh, Daniel Engovatov, Martin Hentschel, Jiansheng Huang, Allison W. Lee, Ashish Motivala, Abdul Q. Munir, Steven Pelley, Peter Povinec, Greg Rahn, Spyridon Triantafyllis, Philipp Unterbrunner (2016)
  Journal: Proceedings of the 2016 International Conference on Management of Data; Publisher: ACM; Pages: 215-226; DOI: [10.1145/2882903.2903746](https://doi.org/10.1145/2882903.2903746)
  详细阐述Snowflake核心架构的学术论文，包括微分区、元数据服务和分布式查询执行。
- [Understanding Snowflake's Unique Architecture](https://docs.snowflake.com/en/user-guide/key-concepts-architecture) — Snowflake Documentation (2024)
  Publisher: Snowflake Inc.
  Snowflake云原生架构的官方指南，说明微分区、数据存储和查询优化。
- [BigQuery architecture overview](https://cloud.google.com/bigquery/docs/bigquery-overview) — Google Cloud Documentation (2024)
  Publisher: Google LLC
  BigQuery存储模型的概述，说明其列式数据、自动分区和基于元数据剪枝方法。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFUmP9rHN5L-RGK_Hew8Ibra3Fr8umvQ4PdSKpYRUeyxE4Pr8HcvmIMofrTrG-q_Bl13tcjioyDCRxqmqQy7RdYtr3qPpvL8Yzs95TgpF79Ch41K1QrI61fpTkdoLnNoG1H9Kk3KDej6H6H5A2XtlOEaIqUtCmUxV5sGT-S1zEt5ww7bc1pMerfdxsmlDc=) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  讨论数据存储策略基本概念、不可变数据结构和分布式系统，这些都与现代数据仓库设计有关。
