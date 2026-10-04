---
course: "building-scalable-data-warehouses"
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-4-query-optimization-performance"
sourceId: 1402
chapter: "query-optimization-performance"
title: "查询优化与性能调优"
order: 4
description: "掌握查询优化：执行计划、修剪、连接策略和物化视图。"
hasQuiz: false
---

在太字节级数据集上执行SQL，不仅仅需要语法正确，它还要求理解底层的分布式机制。当查询在一个大规模并行处理（MPP）系统上运行时，引擎会将声明性逻辑转换为物理执行计划。此计划决定了计算节点如何访问存储层，以及数据在聚合和连接操作期间如何在网络中传输。

在这种环境中，性能瓶颈通常源于过度的I/O或网络拥塞。未能有效使用存储布局的查询，会迫使系统读取不必要的微分区，从而增加成本和延迟。执行分布式查询所需的时间，可以通过对可用节点上的扫描、混洗和处理时间求和来近似计算：

$$ T_{total} \approx \frac{V_{scan}}{R_{disk}} + \frac{V_{shuffle}}{R_{network}} + \frac{I_{cpu}}{P_{nodes}} $$

其中，$V$ 代表数据量，$P$ 代表并行度。减少$V_{scan}$和$V_{shuffle}$是性能调优的主要目标。

在本节中，我们将审视直接影响这些变量的架构决策。我们首先分析查询执行计划，以显示表示查询阶段的有向无环图（DAG）。您将学会识别特定的操作符，例如 `TableScan` 或 `Exchange`，这些操作符表明资源竞争。

接下来，内容转入存储优化技术。我们讨论如何定义聚簇键以实现分区修剪，从而允许引擎跳过不符合查询谓词的数据块。我们还将区分广播连接和混洗连接策略，确定何时在节点间复制维度表，以及何时重新分发事实表。最后，我们使用持久化层，采用物化视图和缓存机制，以避免重复工作负载中的重复计算。
