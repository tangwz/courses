---
course: "data-modeling-schema-design-analytics"
chapter: "physical-design-optimization"
lesson: "partitioning-strategies"
sourceId: 7990
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-5-physical-design-optimization/partitioning-strategies"
title: "分区策略"
description: "通过表分区提升查询性能。"
order: 1
plots: ["plots/7990-0.json"]
sourceHash: "b9439984bb885f1bc3959046f5f0281cd93cf9fe66e76105d5d80bdca5f438a4"
sourceCorrections: []
---

分区在不改变逻辑模式的情况下，改变了磁盘上数据的物理组织。尽管在SQL查询中，表看起来是一个单一连续的对象，但分区会根据特定列的值将该对象分割成独立的段。这种分割使得数据库引擎能够通过一种称为分区剪枝的机制来优化数据检索。

当用户提交带有过滤条件（例如 `WHERE transaction_date = '2023-10-15'`）的查询时，未分区的表会迫使引擎扫描数据集的每个字节以定位匹配的行。在分区表中，引擎会首先查询元数据。它确定哪些物理段包含该特定日期的数据，并严格忽略所有其他段。扫描数据量的减少直接关联到更少的输入/输出 (I/O) 操作和更快的执行时间。

### 分区剪枝的运作方式

分区剪枝依赖于数据库维护每个段的最小值和最大值统计信息。当数据被载入时，系统会将其组织成对应于分区的目录或文件块。

设想一个名为 `web_logs` 的表，它依据 `event_date` 进行分区。物理存储可能看起来像一个目录结构，其中每个日期都是一个包含特定数据文件的文件夹。

> 优化器根据分区元数据评估过滤条件。它仅将扫描指向相关段（绿色），完全绕过不相关段（红色）。

这种操作的效率可以用数学方式描述。如果 $C_{scan}$ 是每字节读取成本，$V_{total}$ 是表的总数据量，则全表扫描的成本简单地是 $C_{scan} \times V_{total}$。通过分区，如果查询定位了分区中的一小部分 $f$（其中 $0 < f < 1$），则成本变为：

$Cost_{分区} \approx C_{scan} \times (V_{total} \times f) + C_{元数据}$

$C_{元数据}$ 项代表检查分区边界所需的小开销。只要跳过的数据量可观，性能提升就显著。

### 选择分区列

选择正确的分区列是此物理设计阶段中决定性的决策。目标是将分区策略与最常见的查询模式相匹配。

#### 基于时间的划分

在分析工作负载中，`WHERE` 子句几乎总是包含一个时间部分。用户通常按天、周或月进行过滤。因此，`date` 或 `timestamp` 列是最常见的分区键。此策略使存储与数据的自然累积对齐 (alignment)，从而更方便地管理生命周期策略，例如归档旧数据或删除不再需要的分区。

#### 列表分区

对于访问模式严格依据类别维度分离的数据集，列表分区是有效的。常见例子包括 `region_id` 或 `country_code`。如果分析师通常运行 `WHERE country = 'FR'` 这样的查询，按国家分区可确保针对法国的查询从不读取属于德国或西班牙的数据。

### 基数与小文件问题

模式设计中一个常见错误是过度分区。当选定的分区具有高基数时（即它包含太多唯一值），就会发生这种情况。

如果你依据 `user_id` 对一个 100 GB 的表进行分区，而你有一百万用户，你将创建一百万个微小分区。这会导致“小文件问题”。分布式存储系统，例如 HDFS 或对象存储（S3、GCS），针对读取大块连续数据进行了优化。当数据被碎片化为成千上万个千字节大小的文件时，元数据开销会超过数据检索时间。

分区数量与查询性能之间的关系通常遵循一条曲线，即性能先改善到某个点，然后由于元数据开销而下降。



![分区粒度对查询延迟的影响](plots/7990-0.json)



> 随着粒度增加（从年到天），延迟降低，因为引擎扫描的数据更少。然而，转向过多的粒度（小时或分钟）会创建太多小文件，导致延迟由于元数据处理开销而再次上升。

为避免这种情况，请遵循以下准则：

1. **目标大小：** 分区大小目标是至少 1 GB。
2. **粗粒度：** 如果你每天只有几兆字节的数据，请按月分区而不是按天分区。
3. **避免高基数：** 从不按具有数千个不同值的列进行分区，除非数据量达到 PB 级别。

### 处理数据倾斜

分区假设数据分布相对均匀。然而，数据通常是倾斜的。设想一个零售系统按 `store_id` 分区。纽约的旗舰店生成的数据可能是一个乡村小镇小型精品店的 100 倍。

这种倾斜导致一个超大型分区和许多小分区。定位纽约门店的查询将不会看到与定位小门店相同的性能优势。此外，并行运行的处理任务可能会遇到“掉队者”，即 99 个任务快速完成，但处理超大型分区的任务会使整个任务持续运行。

在严重倾斜的情况下，你可能需要使用复合分区（按日期分区，然后按区域进行子分区）或依赖数据库的自动聚类能力，我们将在下一节讨论这一点。

### 分区与连接

尽管分区主要优化过滤条件，但如果连接中的两张表都依据连接键分区，它也能帮助提升连接性能。这常被称为“协同定位连接”。

如果一个非常大的事实表和一个大型维度表都依据 `date` 分区，查询引擎可以直接将事实表的分区 A 与维度表的分区 A 连接起来。它知道日期 A 的匹配记录不会存在于日期 B 的分区中。这减少了连接操作所需的内存，因为引擎分段处理连接，而不是将整个数据集载入内存。

然而，大多数星形模式依据日期分区事实表，而维度表通常足够小，可以不分区或进行复制。分区的主要焦点仍然是减少庞大事实表的扫描范围。

## 参考资料

- [Database System Concepts](https://www.db-book.com/db7/) — Abraham Silberschatz, Henry F. Korth, and S. Sudarshan (2021)
  Publisher: McGraw-Hill Education
  理解数据库系统的基础教材，涵盖物理数据组织、索引和查询优化技术，这些是分区的基础。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  数据仓库设计的经典资源，提供构建分析查询数据结构的实用指导，常涉及如针对大型事实表的分区等物理优化。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491920964/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供现代数据系统的广泛视角，涵盖存储、索引和分布式数据处理，有助于理解“小文件问题”和大规模数据组织。
- [Partitioned tables introduction](https://cloud.google.com/bigquery/docs/partitioned-tables) — Google Cloud (2024)
  Google BigQuery 中分区策略的官方文档，包括按时间单位、整数范围和摄取时间分区，并考虑了优化查询性能和管理数据量。
