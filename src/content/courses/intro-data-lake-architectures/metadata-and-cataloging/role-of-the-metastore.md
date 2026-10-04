---
course: "intro-data-lake-architectures"
chapter: "metadata-and-cataloging"
lesson: "role-of-the-metastore"
sourceId: 8011
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-4-metadata-and-cataloging/role-of-the-metastore"
title: "元数据存储库的作用"
description: "元数据存储库如何为查询引擎将物理文件映射到逻辑表。"
order: 1
plots: []
sourceHash: "8a4ec0fca142ebec3670cace1f5743ff0bef7f5082afca0c18893b5961d98053"
sourceCorrections: []
---

像PostgreSQL或Oracle这样的单一数据库系统，其存储引擎和查询引擎紧密关联。系统专门负责数据的写入和读取，因此对数据在磁盘上的确切位置和数据结构了如指掌。

在数据湖架构中，这种关联被打破了。存储由Amazon S3、Google Cloud Storage (GCS) 或Azure Data Lake Storage (ADLS) 等系统负责，而处理则由Apache Spark、Trino或Flink等独立的引擎负责。这些计算引擎是无状态的；它们本身不清楚某个特定的S3存储桶包含客户记录，也不知道该存储桶内文件的模式。

元数据存储库填补了这个空白。它作为数据湖的中央状态存储库，为存储在对象存储中的文件提供持久的结构定义。它允许分布式引擎将一组分散的文件视为可通过SQL访问的结构化表。如果没有元数据存储库，每次查询都将要求用户手动定义模式和文件路径，使大规模的交互式分析难以进行。

### 逻辑与物理层分离

元数据存储库的主要功能是抽象物理存储层。当用户执行 `SELECT * FROM users` 这样的查询时，查询引擎会查询元数据存储库，将逻辑标识符 `users` 解析为物理属性。

元数据存储库为引擎提供以下必要信息：

1. **位置：** 数据存储的基础URI（例如，`s3://my-lake/silver/users/`）。
2. **模式：** 列名、数据类型和顺序。
3. **格式：** 序列化格式（例如，Parquet、Avro、JSON）和压缩编解码器（例如，Snappy、Zstd）。
4. **分区：** 数据如何组织成子目录的定义。

下图描述了元数据存储库如何协调用户SQL请求与原始存储字节之间的交互。

> 在解耦架构中查询执行的流程。引擎在从存储读取任何文件之前，会与元数据存储库交互以定位数据。

### 元数据结构

元数据存储库不存储实际数据（数据集的行和列）。相反，它存储的是*关于*数据的元数据。这通常通过使用关系型数据库后端（如MySQL或PostgreSQL）并通过服务层访问来实现。

当你在数据湖中定义表时，你就是在元数据存储库中创建一个条目。思考以下用于注册数据集的SQL语句：

```sql
CREATE EXTERNAL TABLE sales_data (
    transaction_id STRING,
    amount DOUBLE,
    customer_id INT
)
PARTITIONED BY (transaction_date STRING)
STORED AS PARQUET
LOCATION 's3://enterprise-data-lake/gold/sales/';
```

执行后，元数据存储库会记录特定属性。

**表定义**
元数据存储库为 `sales_data` 创建一个记录，并关联到特定的数据库（命名空间）。它存储模式定义，确保后续查询验证数据类型（例如，确保 `amount` 被视为双精度浮点数）。

**SerDe 信息**
"SerDe" 代表 Serializer/Deserializer（序列化器/反序列化器）。元数据存储库记录查询引擎应使用哪个Java类或库来读取文件。对于上述例子，它注册的是 `org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat`。这个指示确保如果底层文件是Parquet格式，引擎不会尝试将它们读取为CSV或JSON。

**分区索引**
对于分区表，元数据存储库维护着有效分区的索引。如果数据按日期分区，元数据存储库会存有现有分区值的列表（例如，`2023-01-01`、`2023-01-02`）。这允许查询引擎执行“分区剪枝”，跳过不符合查询 `WHERE` 子句的目录扫描。

### 托管表与外部表

在数据湖的背景下，**托管**（或内部）表与**外部**表之间的区别对于数据治理和安全而言值得关注。

**托管表**
当你创建托管表时，元数据存储库控制元数据和数据的生命周期。元数据存储库通常会在默认仓库位置创建一个目录。

- **行为：** 如果你执行 `DROP TABLE`，元数据存储库会删除元数据定义*并*从存储中删除实际文件。

**外部表**
外部表是生产数据湖的标准模式。它们指向你明确管理的特定存储位置。

- **行为：** 如果你执行 `DROP TABLE`，元数据存储库只删除元数据定义。S3或GCS中的底层文件保持不变。

这种分离可以防止意外数据丢失。它还允许多个元数据存储库或不同的计算引擎共享相同底层数据而无冲突。例如，一个Spark作业可能将数据写入一个文件夹，而一个外部表定义则允许独立的商业智能工具读取同一个文件夹。

### Hive元数据存储库标准

Apache Hive元数据存储库（HMS）成为基于Hadoop架构的事实标准，并且仍然是现代数据湖的主导接口协议。即使你没有将Hive用作查询引擎，你也很可能在使用HMS API。

许多云原生目录，例如AWS Glue Data Catalog或Google Dataproc Metastore，都实现了Hive元数据存储库接口。这种兼容性允许Spark、Presto和Trino等引擎与云托管目录交互，就像它们在与传统的Hive元数据存储库通信一样。

然而，传统元数据存储库面临一致性方面的挑战。由于文件系统（S3）和元数据存储库（SQL数据库）是独立的系统，它们可能会失去同步。如果文件直接添加到S3但未在元数据存储库中注册，查询引擎将无法看到它。这一局限性推动了对分区发现机制以及Delta Lake和Iceberg等现代表格式的需求，这些格式将部分元数据管理从元数据存储库移到存储层本身。

### 元数据存储库性能影响

元数据存储库在高并发环境中通常是瓶颈。每个查询规划阶段都需要往返元数据存储库以获取模式和分区列表。

如果一个表有数千个分区，元数据存储库必须序列化并将所有这些位置数据返回给查询引擎。

$T_{\text{规划}} = T_{\text{网络}} + T_{\text{数据库查询}} + T_{\text{序列化}}$

随着分区数量 ($N$) 的增加，$T_{\text{序列化}}$ 线性增长。有效的架构设计涉及限制分区基数或使用高级缓存策略来减少元数据存储库的负担。元数据存储库过载可能导致查询规划超时，即使计算集群空闲且存储系统响应迅速。

## 参考资料

- [Hive Metastore Design](https://cwiki.apache.org/confluence/display/Hive/Hive+Architecture) — Apache Software Foundation (2024)
  介绍Apache Hive Metastore的架构和组件，它是数据湖的中心元数据存储库。
- [What is AWS Glue Data Catalog?](https://docs.aws.amazon.com/glue/latest/dg/components-overview.html#datacatalog-overview) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  官方文档，说明AWS Glue Data Catalog作为与Hive Metastore接口兼容的托管式无服务器元数据存储库的作用和功能。
- [Data Lake Architecture: How to Build a Modern Data Platform for the AI Era](https://www.technicspub.com/data-lake-architecture/) — Bill Inmon, Daniel Linstedt, Ralph Hughes (2020)
  Publisher: Technics Publications
  全面介绍数据湖概念，包括存储与计算的分离，以及元数据管理的基础作用。
