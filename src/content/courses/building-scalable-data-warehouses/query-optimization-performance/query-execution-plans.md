---
course: "building-scalable-data-warehouses"
chapter: "query-optimization-performance"
lesson: "query-execution-plans"
sourceId: 8036
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-4-query-optimization-performance/query-execution-plans"
title: "分析查询执行计划"
description: "解读DAG和执行配置文件以识别性能瓶颈"
order: 1
plots: ["plots/8036-0.json"]
sourceHash: "ba956fd10228878d32ffd91d7b9375c2e8e4e0988e0cc2120e7c882bdb378737"
sourceCorrections: []
---

现代MPP系统中的优化引擎充当编译器，将声明式SQL语句转换为命令式物理执行计划。当查询表现不佳时，执行计划提供明确的诊断线索，以便明白数据库引擎如何解释了你的指令。在大规模应用中，最优计划与次优计划之间的差异很少是线性的。它通常是指数级的，决定了查询是几秒钟完成还是数小时后超时。

执行计划构造为由操作符组成的有向无环图 (DAG)。数据从叶节点（代表存储扫描）流出，经过中间处理节点，最终到达根节点，根节点将结果返回给客户端。在分布式环境中，此图表不仅表示逻辑操作，也表示集群中的物理数据传输。

### 分布式执行计划的构成

理解DAG需要识别操作的层级结构。优化器自下而上构建此树。叶节点访问原始微分区。随着数据在树中向上移动，理想情况下，通过过滤和聚合，数据量会减少。一个良好的计划类似于一个倒置的漏斗。一个不佳的计划通常类似于数据量保持不变的管道，或者中间联接导致行数激增的倒置漏斗。

以下图表展示了分布式表上聚合查询的典型执行流程。请注意“Exchange”操作符的特别引入，它们表示计算节点之间的网络传输。

> 数据从存储通过本地处理，然后经由Exchange操作符进入网络层。

### 识别扫描效率问题

任何计划中的第一个瓶颈通常位于叶节点：`TableScan`。在Snowflake或BigQuery等列式存储中，引擎依靠元数据跳过不含相关数据的微分区。此过程称为分区剪枝。

分析计划时，比较**扫描的分区数**与**总分区数**。如果查询按日期列过滤，但计划表明扫描了100%的分区，则引擎未能利用存储布局。这通常发生在以下情况：

1. 函数包裹了`WHERE`子句中的列（例如，`WHERE TRUNC(event_timestamp) = '2023-01-01'`），这会阻止优化器使用静态元数据。
2. 数据未按过滤列进行聚簇，导致值范围涵盖每个微分区。

扫描的成本主要受I/O限制。我们可以将扫描操作符的延迟贡献 ($L_{scan}$) 表示为：

$L_{scan} = \frac{S_{total} \cdot (1 - P_{factor})}{B_{io}}$

其中 $S_{total}$ 是总表大小，$P_{factor}$ 是剪枝效率（0到1之间），$B_{io}$ 是I/O带宽。当 $P_{factor}$ 为0时，意味着全表扫描，这将使延迟达到最大。

### 交换和洗牌的成本

`Exchange` 操作符是分布式计划中最重要的组成部分。它表示数据正在跨网络传输。在无共享架构中，数据必须物理地位于执行联接或聚合的节点上。

交换主要有两种类型：

- **重新分区（洗牌）：** 行根据键（例如，联接键）进行哈希，并发送到特定节点。如果你联接两个大表，它们通常都会被洗牌，以使匹配的键到达同一个节点。这会消耗大量网络资源。
- **广播：** 联接一侧的整个数据集被复制到集群中的每个节点。这对于小型维度表来说是高效的，但如果应用于大表则可能导致严重问题。

如果你发现`Exchange`操作符消耗了大部分查询时间，请检查进入交换的数据量。如果数百万行数据正在为简单的聚合而洗牌，考虑预先聚合数据，或调整聚簇键使其与联接键对齐 (alignment)，这可能会完全消除洗牌（即实现同地联接）。

### 检测数据倾斜

即使是有效的执行计划，也可能因数据倾斜而表现不佳。优化器假定数据值是均匀分布的。然而，数据通常遵循幂律分布。如果某个`customer_id`占了50%的事务，负责处理该客户的节点将成为“热点”。

执行配置文件通过显示线程或节点间的工作分布来使其形象化。在健康的系统中，所有条形图的高度大致相等。在倾斜的系统中，一个节点在其他节点空闲很久之后仍在继续处理。分布式查询的总运行时间由最慢的节点决定，而非平均节点。

以下图表展示了单个节点成为拖后腿者的倾斜执行配置文件。



![按计算节点划分的执行时间（倾斜检测）](plots/8036-0.json)



> 节点04代表了由数据倾斜导致的拖后腿者，迫使整个集群等待单个进程完成。

### 联接膨胀和基数估计

优化器使用统计数据来估计操作产生的基数（行数）。当这些估计不准确时，就会出现常见的性能问题。

如果计划预测联接将输出10,000行，但实际运行时产生1亿行，下游操作符将被分配不足的内存。这迫使引擎将中间结果溢出到磁盘（本地SSD或远程对象存储）。这种现象，被称为**磁盘溢出**，会导致严重的性能下降，通常是一个数量级。

审查计划时，查看“估计行数”和“实际行数”之间的差异。如果实际计数明显更高，这表示联接逻辑创建了笛卡尔积或多对多关系，从而增大了数据集。这通常意味着联接条件存在语义错误，或者需要在联接之前进行独立的聚合步骤。

### 解读编译器指令

现代优化器并非万无一失。你可能会偶尔看到一些操作符，表明引擎不得不降低其策略。

- **外部排序：** 表示数据集过大，无法在内存中排序。
- **嵌套循环联接：** 虽然对于少量输入来说可以接受，但在大表上看到它，通常表示缺少统计信息，或者存在不等式联接（例如，`t1.a > t2.b`）阻止了哈希联接。

分析执行计划是一个迭代过程。你运行查询，检查DAG中耗时最长的条（时间）或数据量最大的箭头（行），对剪枝或洗牌效率低下形成假设，并相应地重构查询或模式。

## 参考资料

- [Database Management Systems](https://books.google.com/books/about/Database_Management_Systems.html?id=a5mD_qXm_m8C) — Raghu Ramakrishnan, Johannes Gehrke (2002)
  Publisher: McGraw-Hill Education
  为数据库系统中的查询处理、优化和物理计划生成提供了全面的介绍。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  讨论了分布式数据系统的模式，涵盖了并行查询执行和数据分区策略等主题。
- [The Snowflake Elastic Data Warehouse](https://dl.acm.org/doi/10.1145/2882903.2903741) — Benoît Dageville, Thierry Cruanes, Marcin Zukowski, V. N. Antonov, Artin Avanes, Jon Bock, Jonathan Claybaugh, Daniel Engovatov, Martin Hentschel, Jiansheng Huang, Allison W. Lee, Ashish Motivala, Abdul Munir, Steven Pelley, Peter Povinec, Greg Rahn, Spyridon Triantafyllis, Philipp Unterbrunner (2016)
  Journal: Proceedings of the 2016 ACM SIGMOD International Conference on Management of Data; Publisher: Association for Computing Machinery, New York, NY, USA; Pages: 215-226; DOI: [10.1145/2882903.2903741](https://doi.org/10.1145/2882903.2903741)
  详细介绍了云数据仓库Snowflake的架构，重点阐述其查询优化器和执行引擎。
