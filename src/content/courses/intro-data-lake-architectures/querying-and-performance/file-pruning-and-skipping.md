---
course: "intro-data-lake-architectures"
chapter: "querying-and-performance"
lesson: "file-pruning-and-skipping"
sourceId: 8017
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-5-querying-and-performance/file-pruning-and-skipping"
title: "文件修剪与跳过"
description: "查询引擎如何运用元数据和统计信息在执行时忽略不相关的文件。"
order: 2
plots: ["plots/8017-0.json"]
sourceHash: "f8cb1b86c8c0e3dd3c43f402787cd351d8e8116565ff2017d8a2d4c0fb1ffa8a"
sourceCorrections: []
---

查询的性能通常不取决于数据处理的速度，而更多取决于能够完全避免处理多少数据。网络数据传输是解耦存储架构中的主要瓶颈。从对象存储读取的每个字节都会产生延迟和费用。因此，最有效的优化方法是确保查询引擎只获取包含所需行的特定文件。

两种主要机制实现了这种数据量的减少：**分区修剪**和**文件跳过**。尽管它们都旨在减少I/O操作，但它们在查询规划过程中作用的粒度和阶段有所不同。

### 分区修剪

分区修剪是首要的防御措施。它在目录级别（或元数据存储中的逻辑分区级别）起作用。当您按照特定列（如`date`或`region`）定义分区表时，物理数据会被组织成层级目录。

当用户提交一个带有`WHERE`子句、对分区列进行过滤的查询时，查询规划器会检查目录结构或目录中的分区列表。它会立即抛弃任何不符合过滤条件的路径。此操作发生在规划阶段，通常在存储层上列出或访问任何数据文件之前。

设想一个按年和月分区的数据集。如果查询请求`2023年1月`的数据，引擎会有效忽略所有其他年份和月份。

> 查询引擎分离出目标路径`year=2023/month=01`，完全绕过不相关的目录，从而减少元数据开销和扫描。

在这种情况下，引擎不需要列出`year=2022`中的文件。这减少了对对象存储的API调用次数（例如S3的`LIST`请求），并避免引擎为不相关的数据调度任务。

### 文件跳过与统计信息

分区修剪是粗粒度的。它适用于高层次的过滤，但当查询基于非分区列进行过滤时则无效。比如，如果您在`2023年1月`分区内查询`customer_id = 105`，分区修剪会将范围缩小到该月份，但引擎可能仍需扫描该文件夹内的数百个文件。

这就是**文件跳过**（也称数据跳过）作用所在。此技术采用存储在文件页脚（Parquet或ORC格式中）或表清单（Iceberg或Delta Lake格式中）的元数据统计信息。

用于跳过的最常见统计信息包括：

1. **最小值：** 文件中（或行组中）某列的最小数值。
2. **最大值：** 文件中某列的最大数值。
3. **空值计数：** 该列是否包含空值。

#### 跳过如何运作

当查询引擎评估一个文件时，它会查看`WHERE`子句并将其与文件的元数据进行比较。它进行逻辑测试，以判断该文件*是否可能*包含所需数据。

对于过滤条件为`WHERE price > 100`的查询：

- **文件A** (最小: 10, 最大: 50): 范围 [10, 50] 与 > 100 没有重叠。**跳过。**
- **文件B** (最小: 40, 最大: 120): 范围 [40, 120] 与 > 100 有重叠。**读取。**
- **文件C** (最小: 150, 最大: 200): 范围 [150, 200] 完全包含。**读取。**

与下载和解压文件的成本相比，引擎执行此检查所需的CPU资源可以忽略不计。



![基于列范围的文件跳过逻辑](plots/8017-0.json)



> 文件范围的可视化。红线代表一个查询值。只有范围（蓝色条）与查询值相交的文件才会被扫描；其他文件（灰色条）则被跳过。

#### 数据布局的作用

文件跳过完全取决于数据分布。如果您的数据随机分布在各个文件中，那么每个文件的最小值/最大值范围将有效覆盖整个值域。举例来说，如果客户ID从1到1,000,000随机分散，每个文件可能都有接近1的最小值和接近1,000,000的最大值。这种情况下，没有文件可以被跳过，因为每个文件*都可能*包含请求的ID。

为了最大限度提高文件跳过的效率，数据工程师必须在数据摄取时对数据进行聚类或排序。

- **排序：** 通过在写入前对经常查询的列（如`customer_id`）进行数据排序，您可以确保文件1包含ID 1-1000，文件2包含ID 1001-2000，以此类推。这会创建出明确的、不重叠的范围。
- **Z-排序：** 对于基于多列（例如`customer_id`和`transaction_date`）的跳过，简单的线性排序是不够的。Z-排序（或希尔伯特曲线）是一种现代格式（如Delta Lake和Iceberg）使用的技术，用于在多维空间中将相关信息放在一起，从而实现多属性上的高效跳过。

### 开放表格式中的跳过

传统数据湖方法要求查询引擎打开每个Parquet文件的页脚以读取这些统计信息。尽管这比读取整个文件要好，但仍然需要向对象存储发出大量的`GET`请求，仅仅是为了读取文件头。

现代开放表格式（OTFs）如Apache Iceberg通过将统计信息提升到\*\*清单（Manifest）\*\*级别来改进这一点。清单是一个元数据文件，它列出了数据文件及其上下限。

使用OTF时，查询引擎首先读取清单文件。它纯粹在内存中执行过滤逻辑，基于元数据。它生成一个目标文件列表，然后才与对象存储通信以获取实际数据。这种分离使规划阶段与存储层解耦，显著降低了大型表的延迟。

### 效率计算

我们可以采用简化的成本模型来估算修剪和跳过的效果。令 $N$ 为总数据量，$S$ 为引擎的扫描吞吐量 (throughput)。

不进行修剪时，查询时间 $T$ 大致与数据集大小成线性关系：
$T \approx \frac{N}{S}$

通过分区修剪（将数据量过滤至10%）和文件跳过（根据统计信息跳过剩余文件的80%），有效扫描的数据量 $N_{effective}$ 变为：
$N_{effective} = N \times 0.10 \times 0.20 = 0.02N$

查询现在运行速度提高了50倍，不是因为引擎本身更快，而是因为它处理的数据减少了98%。这种减少直接带来了更低的云成本和更快的分析结果。

## 参考资料

- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  介绍 Apache Iceberg 表格式概念，包含清单文件和数据跳过机制。
- [Apache Parquet Format Specification](https://parquet.apache.org/docs/file-format/) — The Apache Software Foundation (2024)
  详细说明 Parquet 文件的结构，包含元数据组织和列统计信息。
- [Delta Lake Documentation](https://docs.delta.io/latest/index.html) — The Linux Foundation (2024)
  涵盖 Delta Lake 性能特性，例如数据跳过和 Z-ordering。
- [Apache Iceberg: A Table Format for Analytic Datasets](https://www.vldb.org/pvldb/vol13/p3178-blue.pdf) — Ryan Blue, Daniel Weeks, Jeremy Klumpp, Russell Spitzer, Fady Essam, Anton Okolnychyi (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 13; Pages: 3178-3191; DOI: [10.14778/3407086.3407153](https://doi.org/10.14778/3407086.3407153)
  介绍 Apache Iceberg 的架构和特性，侧重于其使用清单列表和文件进行元数据管理，以实现高效查询规划和数据跳过。
