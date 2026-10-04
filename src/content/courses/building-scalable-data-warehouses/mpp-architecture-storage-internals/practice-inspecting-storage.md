---
course: "building-scalable-data-warehouses"
chapter: "mpp-architecture-storage-internals"
lesson: "practice-inspecting-storage"
sourceId: 8025
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-1-mpp-architecture-storage-internals/practice-inspecting-storage"
title: "实践：检查存储配置"
description: "实践环节，分析云数据仓库中的存储配置和分区分布。"
order: 5
plots: ["plots/8025-0.json"]
sourceHash: "ad0ae8a0323e5ce546dd58dd895397f29350d5c73e12488d6c9b819186c402f7"
sourceCorrections: []
---

优化MPP系统需要对数据在磁盘上如何存放的了解。查询优化器处理执行路径，而数据的物理排布决定了所需的I/O操作量。检查分布式表的存储元数据可以量化 (quantization)分区修剪的有效性。诊断命令可以评估聚簇状态，并指出逻辑数据顺序与物理存储不一致的情况，从而导致过多的扫描。

### 检查分区指标

当数据摄入Snowflake或BigQuery等系统时，它会被分割成不可变的数据块或微分区。系统为每个数据块维护一个元数据头，其中包含每列的最小和最大值。查询的效率取决于这些最小-最大范围的差异程度。

为此，我们查看**聚簇深度**。该指标表示特定列范围上重叠分区的平均数量。如果表的深度为1，则数据是完全排序的；对该列进行过滤的查询将扫描理论上最少的数据块。高深度表示显著的重叠，迫使引擎扫描多个分区以满足单个谓词。

从数学上讲，如果我们定义一个分区集合 $P$，其中每个分区 $p_i$ 覆盖目标列的范围 $[min_i, max_i]$，那么特定值 $v$ 的重叠计数是：

$Overlap(v) = \sum_{i=1}^{|P|} \mathbb{1}(min_i \leq v \leq max_i)$

整个数据集上的高求和值表明存储效率不佳。

我们可以使用下图直观地了解健康存储配置与碎片化存储配置之间的差异。请注意，在碎片化状态下，重叠的范围如何迫使查询引擎读取三个文件来查找值“50”，而优化状态仅需一个文件。

> 碎片化存储布局与优化存储布局中分区范围重叠的比较。

### 诊断命令与输出分析

在实际环境中，您可以使用系统函数来验证这些配置。本次练习中，我们假定有一个名为`orders_history`的表，其中包含数TB的带时间戳的事务数据。

要在Snowflake中检查存储配置，您需要执行聚簇信息函数。此命令不扫描数据本身，而是读取元数据层，因此执行迅速且成本低廉。

```sql
SELECT SYSTEM$CLUSTERING_INFORMATION('orders_history', '(event_date)');
```

输出返回一个JSON对象，其中详细说明了物理存储的状态。理解此返回对象中的特定字段对于性能调优很必要。

- **total\_partition\_count**: 构成表的微分区总数。
- **average\_overlap\_depth**: 共享相同数据范围的分区的平均数量。值显著大于1表明该表需要重新排序（聚簇）。
- **depth\_histogram**: 一个分布图，显示有多少分区属于特定的重叠区间（例如，0-16，16-32）。

如果您发现直方图中大多数分区处于高深度区间（例如，深度 > 16），那么即使应用了筛选条件，您的查询也很可能在执行“全表扫描”。

### 分析扫描指标

存储配置的最终检验是扫描数据量与返回数据量的比率。我们可以通过执行一个对照查询并检查查询配置文件统计信息来分析这种关系。

考虑一个按单日过滤的查询：

```sql
SELECT * FROM orders_history WHERE event_date = '2023-11-15';
```

在理想的MPP架构中，系统应识别包含`2023-11-15`的特定微分区并忽略其余部分。如果表包含一年的数据，并按日期妥善聚簇，扫描比率应大致为：

$Ratio \approx \frac{1}{365}$

如果查询配置表明100%的分区都被扫描了，那么数据很可能以非顺序方式摄入（例如，历史数据回填与实时流混合），导致元数据的最小-最大指针对于修剪无效。

以下图表显示了聚簇对扫描量随查询选择性变化的影响。请注意，对于高选择性查询（获取小数据片），无序的存储配置会导致I/O成本呈指数级增长。

> 聚簇表和未聚簇表的扫描数据量（GB）与查询选择性百分比对比。



![扫描效率：聚簇 vs 未聚簇](plots/8025-0.json)



### 结果解读

在上面显示的“未聚簇”情况下，I/O曲线几乎是平坦的。这表明请求1%的数据与请求100%的数据成本几乎相同。此行为表示分区修剪失效。

为了解决这个问题，您必须强制重写底层存储。在大多数MPP系统中，这通过以排序方式重写表段的命令来实现。

```sql
-- 手动重新聚簇策略
INSERT OVERWRITE INTO orders_history 
SELECT * FROM orders_history 
ORDER BY event_date;
```

通过在写入操作期间强制执行`ORDER BY`，您可以使物理存储与逻辑查询模式保持一致。执行此操作后，再次运行`SYSTEM$CLUSTERING_INFORMATION`命令应会得到显著降低的`average_overlap_depth`。

需要指出的是，维护完美的存储配置会产生计算成本。每次数据重新排序，系统都会占用资源。因此，您应该只在那些在大量`WHERE`子句或`JOIN`条件中频繁使用的列上定义聚簇键。持续的后台优化（自动聚簇）在某些平台上可用，但必须加以监控，以防维护成本超过查询性能提升所带来的节省。

## 参考资料

- [Clustering Keys and Clustered Tables](https://docs.snowflake.com/en/user-guide/tables-clustering-keys) — Snowflake Documentation (2024)
  Publisher: Snowflake
  介绍了Snowflake的自动聚类、微分区和SYSTEM$CLUSTERING_INFORMATION函数。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本涵盖数据存储、索引、分区策略和分布式系统的综合指南，对优化数据仓库存储非常有用。
- [C-Store: A Column-Oriented DBMS](https://www.vldb.org/conf/2005/papers/p553-stonebraker.pdf) — Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cherniack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel R. Madden, Elizabeth J. O'Neil, Patrick E. O'Neil, Alexander Rasin, Nga Tran, Stan B. Zdonik (2005)
  Journal: Proceedings of the 31st International Conference on Very Large Data Bases (VLDB); Publisher: VLDB Endowment; Pages: 553-564
  介绍了列式数据库管理系统的基础架构，详细阐述了存储结构和查询处理，对于理解现代MPP数据仓库至关重要。
- [Introduction to BigQuery storage](https://cloud.google.com/bigquery/docs/overview-storage) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  描述了BigQuery的列式存储格式、物理布局以及分区和聚类如何提升查询性能。
