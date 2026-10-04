---
course: "intro-data-lake-architectures"
chapter: "file-formats-and-optimization"
lesson: "open-table-formats"
sourceId: 8001
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-2-file-formats-and-optimization/open-table-formats"
title: "开放表格式"
description: "介绍为数据湖带来ACID事务和时间旅行的表格式。"
order: 3
plots: ["plots/8001-0.json"]
sourceHash: "45b4243aa1ac3bf093680c324f4ad929710b2e7c91368d2ca2b9a2c476928400"
sourceCorrections: []
---

原始对象存储结合Parquet等高性能文件格式，有效解决了可扩展、低成本存储的问题。但与传统关系型数据库相比，这种组合存在一个显著不足：缺少事务保证。当你对一个包含Parquet文件的文件夹执行查询时，查询引擎只会读取该目录下现有的文件。如果数据管道在写入一批文件的中途失败，或者两个用户试图同时修改数据集，查询引擎可能会返回不完整或损坏的结果。

开放表格式（OTFs）通过在原始数据文件之上添加元数据层来弥补这一不足。系统不再将目录视为一个表，而是依赖于一个集中的日志或元数据树来规定在任何特定时刻哪些文件属于该表。这种方法使得数据湖上能够进行ACID（原子性、一致性、隔离性、持久性）事务。

### 元数据层

在标准数据湖中，目录结构定义了数据集。而在采用开放表格式的架构中，表的状况由特定的元数据文件定义。当查询引擎请求数据时，它首先读取元数据以找到有效的数据文件列表。

这种解耦使得原子操作成为可能。当摄入新一批数据时，系统会将新的Parquet文件写入存储，但这些文件对读取者而言是不可见的，直到新的元数据文件被提交。此提交操作是原子的；它将指针从旧的元数据版本切换到新的版本。读取者看到的是完整的旧状态或完整的完整新状态，绝不会是部分写入。

该结构通常遵循分层模式。

> 开放表格式的分层结构，说明了目录指针如何最终指向单个数据文件。

这些组件的功能如下：

1. **目录指针：** 告知引擎表的当前“头部”在哪里。
2. **元数据文件：** 包含表级别信息，如表结构、分区定义和快照列表。
3. **快照：** 表示表在特定时间点的状态。
4. **清单列表：** 构成一个快照的清单文件列表。
5. **清单文件：** 实际数据文件（例如 `s3://bucket/data/file_a.parquet`）列表，以及该文件中列的统计信息，如最小值/最大值。

### ACID事务与并发

此架构的主要优点是在分布式存储中实现了ACID事务。

**原子性**通过元数据交换实现。如果一个`INSERT`作业写入了100个文件中的99个后崩溃，元数据指针将永远不会更新。这99个“孤立”文件存在于存储中，但由于它们未被活动快照引用，因此会被查询忽略。

**隔离性**通常通过快照隔离来处理。当一个查询开始时，它会锁定到特定的快照ID。即使摄入作业在几毫秒后提交了新数据，正在运行的查询仍会继续从它最初启动的快照中读取。这保证了长时间运行的分析查询结果的一致性。

**并发**控制机制在不同格式间有所不同，但通常基于乐观并发控制（OCC）。如果两个写入者试图同时更新表，第一个提交的会成功。第二个写入者会检测到底层版本已改变，必须重试操作或失败，这根据冲突解决策略而定。

### 时间旅行与回滚

由于开放表格式（OTFs）会追踪快照的历史，它们使得“时间旅行”功能成为可能。你可以查询数据在特定时间戳或快照ID时的状态。这对于复现机器学习 (machine learning)模型或审计变更特别有用。

要使用SQL查询表的先前版本，语法通常如下所示：

```sql
-- 查询特定时间点的数据
SELECT * FROM events FOR SYSTEM_TIME AS OF '2023-10-27 10:00:00';

-- 查询特定版本ID的数据
SELECT * FROM events FOR SYSTEM_VERSION AS OF 123456789;
```

此机制也使得即时回滚成为可能。如果数据工程师不小心部署了损坏的管道，恢复表不是通过恢复备份来完成。它仅仅是一个元数据操作，将“当前”引用指回先前的快照ID。

### 比较主要格式

三种主要规范主导着这个生态系统：Apache Iceberg、Delta Lake和Apache Hudi。它们有相同的目标，但其内部实现侧重于不同的工作负载。

#### Apache Iceberg

Iceberg侧重于大规模数据集上的高性能查询。其设计强调物理文件布局与逻辑表结构的分离。一个特点是**隐藏分区**。在传统的Hive风格分区中，如果数据按天分区，用户必须查询`WHERE date_str = '2023-01-01'`。如果他们按时间戳查询`WHERE event_time > '2023-01-01 00:00:00'`，引擎可能无法识别分区关系并扫描整个表。Iceberg保持了列与分区转换之间的关联，使得引擎无论查询如何编写都能正确修剪分区。

#### Delta Lake

Delta Lake最初由Databricks开发，它采用事务日志（`_delta_log`文件夹），其中包含记录顺序原子提交的JSON文件。其功能类似于关系型数据库中的预写式日志（WAL）。Delta Lake主要依赖于文件系统原子性或独立的协调服务（例如AWS S3上的DynamoDB）来保证线性历史。它针对读取性能进行了高度优化，并与Spark紧密结合。

#### Apache Hudi

Hudi（Hadoop Upserts Deletes and Incrementals）由Uber构建，用于处理流式数据摄入和可变数据集。它在需要频繁更新和删除的场景中表现出色。Hudi引入了不同存储类型：

- **写时复制（COW）：** 数据以列式格式（Parquet）存储。更新会重写文件。更适合读密集型工作负载。
- **读时合并（MOR）：** 数据以列式（Parquet）和行式（Avro）日志文件的组合形式存储。更新会追加到日志中，并在查询时合并。更适合写密集型流式工作负载。

COW和MOR之间的选择体现了在写入延迟和读取延迟之间的一种权衡。



![延迟权衡：写时复制与读时合并](plots/8001-0.json)



> 性能特点表明，与写时复制相比，读时合并针对更快的写入进行了优化，但代价是读取速度较慢。

### 小文件管理与压缩

流式摄入中的一个常见问题是“小文件问题”。如果一个流每隔几秒写入一条记录，你最终可能会产生数百万个千字节大小的Parquet文件。这会大幅降低读取性能，因为查询引擎打开和关闭文件所花费的时间比读取数据的时间更多。

开放表格式（OTFs）提供原生压缩工具（常被称为“bin-packing”）。后台进程会读取这些小文件，并将它们重写为更大、理想的文件（通常为128MB到1GB），而不会阻塞读取者。然后元数据会更新，指向新的大文件，并将小文件标记 (token)为逻辑删除。

此过程是幂等的，且对最终用户不可见。压缩成本$C$通常与写入的数据量成正比，但读取时间$R$的节省量与减少的文件数量呈指数级关联。

$\text{读取效率} \propto \frac{1}{\text{文件数量}}$

通过严格管理文件布局和元数据，开放表格式将数据湖从一个静态文件存储库转变为一个能够支持可靠企业分析的动态事务型数据库。

## 参考资料

- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  详细介绍Iceberg的架构，包括其元数据层和隐藏分区等功能。
- [Delta Lake Documentation](https://docs.delta.io/latest/index.html) — The Linux Foundation (2024)
  阐述Delta Lake的设计、事务保证以及时间旅行和模式演变等功能。
- [Apache Hudi Documentation](https://hudi.apache.org/docs/latest/) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  全面解释Hudi处理可变数据集、流式摄取和存储类型（COW、MOR）的方法。
- [Delta Lake: High-Performance ACID Table Storage for Big Data](https://www.vldb.org/pvldb/vol13/p3042-lee.pdf) — Denny Lee, Tathagata Das, Michael Armbrust, Burak Yavuz, Xiangrui Meng, Shixiong Zhu, Joseph Bradley, Brooke Wenig, Ali Ghodsi, and Matei Zaharia (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 13; Pages: 3042-3054; DOI: [10.14778/3407612.3407647](https://doi.org/10.14778/3407612.3407647)
  一篇学术论文，描述了Delta Lake的设计和性能特征，说明了如何为数据湖添加事务能力。
