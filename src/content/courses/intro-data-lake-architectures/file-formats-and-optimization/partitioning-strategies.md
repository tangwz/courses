---
course: "intro-data-lake-architectures"
chapter: "file-formats-and-optimization"
lesson: "partitioning-strategies"
sourceId: 8003
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-2-file-formats-and-optimization/partitioning-strategies"
title: "分区策略"
description: "设计分区方案以减少数据扫描并避免倾斜。"
order: 5
plots: ["plots/8003-0.json"]
sourceHash: "7375b89610db7801f94247076d149de92b47543e56c27394e7a31746bde8f81b"
sourceCorrections: []
---

物理文件布局直接影响数据湖的输入/输出 (I/O) 效率。虽然 Parquet 等列式格式优化了单个文件内数据的读取方式，但分区则优化了最初要打开哪些文件。没有分区，查询引擎必须扫描数据集中的每个文件才能找到匹配记录。采用有效的分区策略，引擎可以完全跳过不相关的文件，这个过程称为分区修剪。

### 目录结构作为索引

在传统关系数据库中，索引是与表一起存储的独立数据结构（如 B 树）。在采用对象存储（S3、Azure Blob、GCS）的数据湖中，目录结构本身充当主要索引。这被称为 Hive 风格分区。

当您将数据帧写入按特定列（例如 `date`）分区的数据湖时，写入器会将文件组织成层次目录。每个目录代表该列的一个不同值。

考虑一个服务器日志数据集。如果您将这些数据存储为平面结构，过滤特定日期的查询需要列出并读取所有对象。但是，如果您按 `date` 分区，结构会像这样：

`s3://bucket/logs/date=2023-01-01/part-001.parquet`
`s3://bucket/logs/date=2023-01-02/part-001.parquet`

当查询引擎执行 `SELECT * FROM logs WHERE date = '2023-01-01'` 时，它会确定只需要扫描目录 `date=2023-01-01`。它实际上忽略了存储桶的其余部分。I/O 吞吐量 (throughput)的这种降低最大程度地减少了延迟并降低了成本，因为云服务提供商对数据扫描和 API 列表请求都收费。

下图说明了查询引擎在读取操作期间如何与分区目录结构交互。

> 分区修剪逻辑，查询引擎根据 WHERE 子句定位特定目录，避免在欧盟和亚太分区上进行不必要的 I/O 操作。

### 基数与小文件问题

选择正确的分区列需要在查询性能和文件管理开销之间取得平衡。这个决策的首要考虑是基数，即列中唯一值的数量。

#### 高基数

按高基数列（例如 `user_id` 或 `transaction_id`）分区通常会降低性能。如果您有 1000 万用户并按 `user_id` 分区，写入过程可能会生成 1000 万个小目录，每个目录包含微小文件（通常只有几千字节）。

这会引发“小文件问题”。Spark 或 Trino 等分布式查询引擎每次打开文件都会产生开销。如果引擎必须打开 10,000 个文件才能读取 1 GB 数据，则列出元数据和建立连接所花费的时间会超过实际读取数据所花费的时间。此外，文件系统和对象存储对元数据操作（列出对象）的限制远早于对带宽的限制。

#### 低基数

按基数极低的列（例如 `status`（活跃/非活跃））分区，可能导致文件过大或分区过宽，无法提供有效的修剪。如果一个分区包含 90% 的数据，修剪带来的好处微乎其微。

#### 理想平衡点

有效的分区列通常将数据分成可管理的小块，其中生成的文件大小在 128 MB 到 1 GB 之间。常见选择包括：

- **基于时间的：** `date`、`year`、`month`（用于时间序列数据）。
- **分类的：** `region`、`department`、`platform`。

如果您的数据量较低（例如，总计低于 100 GB），分区可能会带来比好处更多的开销。在这种情况下，依靠 Parquet 文件的内部行组通常已足够。

### 数据倾斜

数据倾斜发生在数据在分区之间分布不均匀时。在许多数据集中，某些类别占据主导地位。例如，如果您按 `log_level` 分区日志，`INFO` 分区可能包含数 TB 的数据，而 `FATAL` 仅包含数 MB。

处理倾斜数据时，分配给较大分区的并行工作进程完成任务所需的时间明显长于分配给较小分区的进程。这种被称为“拖慢者”的现象，导致整个作业等待最慢的任务完成。

下面的可视化图表强调了按倾斜列分区时分布不均匀的影响。



![数据倾斜对分区大小的影响](plots/8003-0.json)



> 此分布图显示了严重的倾斜，其中“US”分区明显大于其他分区。处理此分区成为分布式计算作业的瓶颈。

为了缓解倾斜，工程师通常使用派生列或分桶。例如，您可以不单单按 `country` 分区，而是按 `country` 和 `user_id` 的哈希桶进行分区。

### 显式分区与隐藏分区

在上述 Hive 风格的方法中，物理目录结构（`date=2023-01-01`）与逻辑表定义紧密耦合。这迫使用户需要了解物理布局才能编写高效的查询。如果用户编写 `WHERE timestamp > '2023-01-01T00:00:00'`，引擎可能不会触发分区修剪，因为物理列是 `date`，而不是 `timestamp`。

Apache Iceberg 等现代开放表格式引入了**隐藏分区**的机制。在 Iceberg 中，您可以定义一个分区转换（例如 `days(timestamp)`）。表格式处理查询中 `timestamp` 列与底层每日分区文件之间的映射。用户查询逻辑列，引擎自动应用必要的修剪策略，而无需用户了解具体的目录布局。

### 分区策略检查表

在设计新表布局时，请使用以下参数 (parameter)来验证您的策略：

1. **谓词分析：** 审查分析查询中最常见的 `WHERE` 子句。分区列必须与这些过滤器匹配才能生效。
2. **文件大小目标：** 确保分区后，平均文件大小保持在 128 MB 到 1 GB 左右。如果文件降至 64 MB 以下，您的分区粒度可能过细。
3. **分区深度：** 避免创建超过 3 或 4 层深度的目录结构（例如，`year/month/day/hour`）。过多的嵌套会使元数据管理复杂化。
4. **基数限制：** 一般而言，表中的分区总数应保持可管理（通常根据元存储能力，在 10,000 到 50,000 之间）。

通过严格遵循这些原则，您可以确保存储层高效地服务于计算层，从而降低分析查询的成本和持续时间。

## 参考资料

- [Apache Iceberg: An Open Table Format for Large-Scale Analytic Datasets](https://www.vldb.org/pvldb/vol11/p1642-blue.pdf) — Ryan Blue, Daniel Weeks, Jeremy Klop, and Jason Zhang (2018)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment Inc.; Volume: 11; Pages: 1642-1653; DOI: [10.14778/3275727.3275730](https://doi.org/10.14778/3275727.3275730)
  描述了 Apache Iceberg 的设计和优势，包括其隐藏分区和模式演进方法，解决了传统 Hive 风格分区的局限性。
- [Apache Hive Documentation - Partitioned Tables](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQETzLIoveh8vo36_MYFCXtuumlsUnWAeFxuJJ1j1wvD666tKXY8eW5CHwnvVcQYQvcJTNLIVLxDdd_mbGaGl_VvlPQJeTCQt45iTnVXX2ibwuLUNWExNND4n49ymZ91bQjRvVQtN0KJG-n9KshCrZVHc_rbMEfncocEKhlN2og==) — The Apache Hive Community (2023)
  Publisher: Apache Software Foundation
  解释了传统 Hive 风格分区的机制和优点，这是数据湖中基于目录的索引的基础。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGfD7Kdd-HBeB3ohtBR75JTUSZ8GqQt3pGdfSygNXrnzz1j2Gdv3ibbtxGLJUrcEw1asY19eT6nE1TvN2wL5pDKm_HRK63GkuMezj7NTImCwkO4grQNKAP7u6tnYuzvOJ1Fr141VqeJhYHWgh0rZuMBJbD-UC4Yu3scM_CfbL7py9Ul5gzWRER1R2ZBT18=) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本广泛认可的著作，涵盖了数据存储、索引和分布式系统的基本概念，为数据湖中分区策略的设计选择提供了背景。
- [Optimizing Amazon S3 performance for data lakes](https://docs.aws.amazon.com/whitepapers/latest/s3-performance-best-practices/s3-performance-best-practices.pdf) — Amazon Web Services (2019)
  Publisher: AWS
  讨论了在 Amazon S3 上优化数据湖性能的实用策略，包括分区的重要性、文件大小确定和其他架构考量。
