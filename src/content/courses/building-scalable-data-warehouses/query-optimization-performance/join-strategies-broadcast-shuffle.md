---
course: "building-scalable-data-warehouses"
chapter: "query-optimization-performance"
lesson: "join-strategies-broadcast-shuffle"
sourceId: 8038
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-4-query-optimization-performance/join-strategies-broadcast-shuffle"
title: "连接策略：广播连接与洗牌连接"
description: "了解分布式连接算法和网络数据传输开销。"
order: 3
plots: ["plots/8038-0.json"]
sourceHash: "1f37c49660c4b53e941197bf671df63dab97e77265390a60529543298268abc3"
sourceCorrections: []
---

分布式连接是MPP环境中资源消耗最大的操作。与单节点数据库（其性能通常由内存带宽决定）不同，分布式连接严格受限于网络吞吐量 (throughput)。在无共享架构中，连接条件所需的数据通常分布在不同的计算节点上。为满足连接谓词，系统必须在集群中物理传输数据，以将匹配的键对齐 (alignment)到同一个节点。

这种数据传输被称为“交换”或“洗牌”阶段，会引入延迟，且延迟会随着数据量和集群规模的增大而线性增长。查询优化器必须选择一种策略，在最大程度减少网络流量的同时，防止单个节点的内存溢出。分布式连接的两种主要算法是洗牌连接和广播连接。

### 洗牌连接策略

洗牌连接是连接两个大数据集的默认机制。它基于哈希分区原理工作。为在 `id` 列上连接表 A 和表 B，引擎会对两表中每行的 `id` 列应用哈希函数。具有相同哈希值的行会被路由到同一个计算节点。

这种策略是对称的。连接关系的两侧都参与网络传输。如果在一个 10 TB 的 `orders` 表和一个 2 TB 的 `line_items` 表之间执行连接，系统会在网络中重新分布 12 TB 的数据（假设在连接前没有发生任何过滤）。

下图显示了洗牌连接过程中数据的流动。请注意两表的部分数据如何通过网络传输，以到达它们被分配的分区桶。

> 一种数据再分布模式，其中两个输入表都被哈希处理并跨集群传输，以对齐 (alignment)连接键。

洗牌连接具有很高的扩展性，因为每个节点的内存需求由分区桶的大小决定，而不是总表大小。然而，它们会产生大量的网络开销。如果网络拥塞，查询计划中的 `Exchange` 操作符将占据总执行时间的主要部分。

### 广播连接策略

广播连接提供了一种替代方法，这种方法针对一个表远小于另一个表的场景进行了优化。这种模式在星型模式查询中很常见，比如一个非常大的事实表与一个小的维度表连接（例如，`orders` 表连接 `currency_codes` 表）。

不同于对两个表进行哈希处理，引擎会将较小的表（构建端）复制到每个持有较大表（探测端）一部分数据的计算节点上。较大的表保持不动。这完全消除了大型事实表的网络传输。

假设连接一个 10 TB 的 `orders` 表（分布在 10 个节点上）和一个 50 MB 的 `currency_codes` 表。

1. **洗牌连接：** 在网络中传输约 10 TB + 50 MB。
2. **广播连接：** 在网络中传输 50 MB $\times$ 10 个节点 = 500 MB。

网络流量的减少很显著。然而，这种策略受限于内存。每个节点都必须将整个广播的表加载到内存中，以高效执行连接。如果维度表超过可用内存，查询可能会溢出到磁盘或因内存不足 (OOM) 错误而失败。

> 一种复制机制，其中较小的维度表被复制到所有包含较大事实表分区的节点。

### 成本分析和选择标准

现代查询优化器 (CBOs) 会估算每种策略的成本，以选择最高效的路径。这个决定很大程度上依赖于表统计信息，具体来说是基数和字节大小。

洗牌连接的网络成本 ($C_{shuffle}$) 大致是两个关系大小的总和：

$C_{	ext{洗牌}} \approx \text{大小(表A)} + \text{大小(表B)}$

广播连接的网络成本 ($C_{broadcast}$) 是小表的大小乘以工作节点数量 ($N$)：

$C_{	ext{广播}} \approx \text{大小(小表)} \times N_{	ext{节点}}$

仅当满足以下条件时才优先选择广播策略：

$\text{大小(小表)} \times N_{\text{节点}} < \text{大小(表A)} + \text{大小(小表)}$

由于 $Size(Small\_Table)$ 相对于 $Table_A$ 可以忽略不计，这就简化为检查广播开销是否小于洗牌大表的开销。

存在一个临界点，此时“小”表变得过大，无法高效广播。随着集群规模的扩大（$N_{\text{节点}}$ 增加），广播的成本线性增加。一个在 4 节点集群上能高效广播的表，在 100 节点集群上可能会导致性能下降。

下图显示了随着维度表大小的增加，洗牌连接变得比广播连接更高效的交叉点。



![网络开销：广播连接对比洗牌连接](plots/8038-0.json)



> 对比分析显示了对于一个静态事实表，广播成本如何随表大小变化，相对于固定的洗牌成本。

### 在执行计划中识别策略

你可以通过检查查询配置文件来识别引擎使用了哪种策略。

1. **Snowflake：** 在配置文件中寻找 `Join` 操作符。如果连接显示“构建侧较小”或在连接前显示 `Broadcast` 交换操作符，则表示它使用了广播策略。`Repartition` 或 `Exchange` 操作符表示洗牌连接。
2. **Spark/Databricks：** 用户界面会明确地将交换标记 (token)为 `BroadcastExchange` 或 `ShuffleExchange`。
3. **BigQuery：** 执行阶段会为洗牌操作显示 `Repartitions`。带有 `Broadcast` 输入的阶段意味着广播策略。

### 数据倾斜和性能影响

洗牌连接中一个潜在的风险是**数据倾斜**。哈希分区假设数据在节点间均匀分布。然而，如果某个特定的连接键（例如 `NULL` 值或通用的“访客”用户 ID）出现频率异常高，则所有匹配该键的行都会集中到一个节点上。

这会导致一个“落后者”任务。一个节点的工作量远大于其他节点，导致整个查询等待该单个节点完成。

- **洗牌倾斜：** 如果 `Table A` 有 10 亿行，其中 2 亿行的 `customer_id` 为 `NULL`，则在洗牌过程中，一个节点将接收 2 亿行，导致内存溢出。
- **广播免疫：** 广播连接通常不受探测（大）侧倾斜的影响，因为数据保持不变。大表的分布不会改变；只复制了小表。

### 优化指导

尽管 CBOs 运行精密，但它们依赖的统计信息可能过时或缺失。当引擎做出次优选择时，你可以使用优化器提示来影响连接策略。

- **强制广播：** 当你确定一个表适合内存，但优化器错误地高估了基数时使用此选项。
  - *SQL 示例：* `SELECT /*+ BROADCAST(d) */ * FROM fact f JOIN dim d ON f.id = d.id`
- **强制洗牌：** 如果广播导致内存不足 (OOM) 错误（因为复制的表对于节点的内存堆来说稍大）时使用此选项。
  - *SQL 示例：* `SELECT /*+ SHUFFLE(d) */ ...`（语法因方言而异）。

了解数据的物理传输可以帮助你预测查询性能。调整高延迟查询时，检查连接输入。如果一个巨大的表正在不必要地进行洗牌，或者一个中等大小的表正在广播并造成内存压力，明确的提示或模式调整（例如改进集群）是优化的主要手段。

## 参考资料

- [Database Management Systems](https://books.google.com/books/about/Database_Management_Systems.html?id=z1IqAQAAIAAJ) — Raghu Ramakrishnan, Johannes Gehrke (2003)
  Publisher: McGraw-Hill Education
  一本标准教科书，提供了分布式数据库和查询处理的基本概念，包括连接算法。
- [Performance Tuning Guide - Join Strategy Hints](https://spark.apache.org/docs/latest/sql-performance-tuning.html#join-strategy-hints) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  Apache Spark SQL 的官方文档，解释了连接策略，包括广播和混洗连接，以及如何对其进行干预。
- [Query Profile](https://docs.snowflake.com/en/user-guide/ui-query-profile) — Snowflake Inc. (2025)
  Publisher: Snowflake Inc.
  Snowflake 的官方文档，描述了如何分析查询执行计划以理解连接策略和性能。
- [Skew-Tolerant Join Methods for MapReduce](https://dl.acm.org/doi/10.14778/340270.340277) — Bugra Okcan, Vadim Riabov (2011)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 4; Pages: 1109-1120; DOI: [10.14778/340270.340277](https://doi.org/10.14778/340270.340277)
  这篇论文讨论了在大型数据处理系统中，分布式连接操作中处理数据倾斜的方法。
