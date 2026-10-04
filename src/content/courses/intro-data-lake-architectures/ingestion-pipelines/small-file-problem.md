---
course: "intro-data-lake-architectures"
chapter: "ingestion-pipelines"
lesson: "small-file-problem"
sourceId: 8008
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-3-ingestion-pipelines/small-file-problem"
title: "小文件问题"
description: "通过文件合并缓解因过多小文件导致的性能下降。"
order: 4
plots: ["plots/8008-0.json"]
sourceHash: "528572d95614fde28c01288086fb59e0da25c72a69a35836e351d83ae122a4c9"
sourceCorrections: []
---

为提高速度而设计的摄取管道常常无意中通过生成过多小文件来降低读取性能。这种通常被称为“小文件问题”的现象，发生于数据写入存储时的增量远小于底层文件系统或对象存储的最佳块大小时。尽管单次写入很快成功，但数千甚至数百万小文件的累积会给后续的分析查询带来显著的性能瓶颈。

### 延迟与吞吐量 (throughput)的运作方式

要弄清楚小文件为何会严重影响性能，我们必须考察分布式查询引擎如何与Amazon S3、Azure Blob Storage或Google Cloud Storage等对象存储系统配合。

对象存储针对高吞吐量（持续读取大量数据）进行了优化，而非低延迟（处理许多小型请求）。当像Trino或Spark这样的查询引擎读取数据集时，它会执行两种不同类型的操作：

1. **元数据操作：** 列出存储桶中的对象以确定要读取哪些文件 (`LIST`)，并获取文件统计信息 (`HEAD`)。
2. **数据传输：** 建立连接并流式传输字节数据 (`GET`)。

每个文件都会产生元数据操作和连接建立的固定开销。这种开销与文件大小无关，是固定值。如果我们对读取大小为 $S$ 并分成 $N$ 个文件的数据集所需的总时间 $T$ 建模，关系如下：

$T_{总} = N \times (T_{开销}) + \frac{S}{带宽}$

当 $N$ 较小（文件较大）时，带宽项占主要地位，系统高效运行。当 $N$ 较大（文件较小）时，$N \times T_{开销}$ 项占主要地位。系统花费更多时间等待服务器响应和管理连接，而不是实际传输数据。

考虑相同1GB数据集的两种文件布局在处理开销上的差异。



![读取时间：开销与数据传输](plots/8008-0.json)



> 1GB数据估算读取时间的比较。对于大文件，开销微不足道，但当数据碎片化时，开销就成为主要瓶颈。

### 摄取管道中的常见原因

小文件问题通常源于摄取阶段的Bronze（原始）层。导致此问题的主要架构模式有两种。

#### 高频流式传输

实时摄取管道通常采用微批处理。如果一个Spark Structured Streaming作业每60秒运行一次触发，它每分钟会向存储层提交一个新文件。在24小时内，单个流会生成1,440个文件。如果数据量较低（例如，每小时10MB），则每个文件仅约7KB。将此扩展到100个并发流，每天将产生近150,000个小文件。

#### 过度分区

工程师经常过度分区数据，以优化特定的查询过滤器。例如，按`year`、`month`、`day`和`hour`对数据集进行分区，会创建一个目录结构，每天将数据分隔到24个文件夹中。如果数据再按高基数列（如`sensor_id`）分区，数据就会过度分散。

如果一个摄取作业写入1GB数据，但将其分散到1,000个分区目录中，平均文件大小将降至1MB。查询引擎随后必须列出所有1,000个目录以重构数据集，这明显增加了查询的规划阶段时间。

### 对目录同步的影响

对于查询性能而言，小文件会对元数据目录（如Hive Metastore或AWS Glue）产生负面影响。目录必须追踪每个文件的位置和模式。文件数量激增会使元数据数据库膨胀，导致：

1. 分区发现时间变慢。
2. 更新表统计信息时出现超时。
3. 对于根据存储对象数量收费的托管目录服务，费用增加。

### 缓解策略：文件合并

小文件问题的标准解决方案是**文件合并**。此过程涉及读取一系列小文件，然后将它们重写为数量更少、大小更大的文件，以提高读取性能。

文件合并通常作为维护作业实施，与摄取管道异步运行。这会将低延迟写入要求（本身会产生小文件）与高性能读取要求分离。

这种架构遵循“快速写入，稍后优化”的模式。

> 将低延迟摄取与文件布局优化分离的工作流程。文件合并作业将原始文件合并成读取优化的块。

#### 装箱算法

文件合并作业通常采用“装箱”算法。其目的是合并文件，直到它们的大小总和达到目标阈值（对于Parquet文件，通常在128MB到1GB之间）。

如果您有十个10MB的文件，目标大小为100MB，装箱算法会将它们分组为一个写入任务。这可以尽量减少重写过程中的网络通信量，并确保生成的文件大小适合向量 (vector)化和压缩。

### 使用开放表格式的实现

现代开放表格式，如Apache Iceberg和Delta Lake，抽象化了文件管理的繁琐之处。它们提供内置程序来处理文件合并，无需编写自定义的文件列表逻辑。

在没有表格式的标准数据湖中，文件合并需要谨慎的编排以确保数据一致性。您必须读取小文件，写入新的大文件，然后原子性地交换它们或更新元数据指针。如果作业中途失败，您会面临数据重复或丢失的风险。

表格式通过使用快照隔离来解决这个问题。文件合并作业可以在后台重写文件，而读取器继续查询旧快照。一旦重写完成，表格式会原子性地提交一个指向大文件的新快照。

例如，在Delta Lake中，`OPTIMIZE`命令执行此功能：

```sql
-- Delta Lake中用于合并文件的标准SQL命令
OPTIMIZE events_table
WHERE date >= current_date() - 1
ZORDER BY (event_type);
```

这条简单的命令会触发一个作业，扫描`events_table`，识别大小低于阈值的文件并重写它们。可选的`ZORDER BY`子句通过将相似数据共同放置在同一组文件中，进一步提高性能，其作用类似于多维度排序。

### 摄取设计的最佳实践

为维护健康的数据湖，请应用这些设计原则：

1. **调整缓冲区大小：** 在Kafka Connect或Kinesis Firehose等流处理框架中，配置缓冲提示。如果延迟要求允许，在刷新到存储之前，让缓冲区至少填充到64MB，或等待更长的时间间隔（例如5-15分钟）。
2. **避免过度分区：** 除非数据生命周期管理确实需要，否则不要按高基数列（如`user_id`或`transaction_id`）分区。优先按更粗的粒度（如`date`）分区，并依靠文件跳过（最小/最大统计信息）进行细粒度过滤。
3. **定期安排维护：** 如果您无法在源头缓冲数据，请安排文件合并作业在批量加载后立即运行，或定期为流式数据运行。
4. **监控文件数量：** 设置可观测性警报。如果表中平均文件大小降至10MB以下，请触发警报以检查摄取配置。

## 参考资料

- [Amazon S3 Performance Guidelines](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html) — Amazon Web Services (2023)
  Amazon S3 中优化数据传输和请求性能的建议，包括并发和文件大小的考量。
- [Spark Documentation: Performance Tuning](https://spark.apache.org/docs/latest/tuning.html) — The Apache Software Foundation (2023)
  Spark 应用程序性能优化指南，涵盖 I/O、数据序列化和内存管理。
- [Delta Lake Documentation: OPTIMIZE](https://docs.delta.io/latest/optimizations.html) — The Delta Lake Project (2023)
  Delta Lake `OPTIMIZE` 命令在数据压缩和 Z-ordering 方面的说明。
- [Apache Iceberg Documentation: Table Maintenance](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPNLUpQlni4asLx1JGF5nz-kjAcEa3TNUBCnm1DLpfaqyq-reveqofCDnyUgyAQ3Cfb5ZqJ4HNOywppzlHkul4G-2NTMg9mdhimVRLFvgtcjLwJNl2xRr5Z_Al63pxHiB-Wj6MviF-CmvXqWo=) — The Apache Software Foundation (2025)
  Publisher: The Apache Software Foundation
  Iceberg 表维护操作指南，包括压缩、清单清理和重写功能。
