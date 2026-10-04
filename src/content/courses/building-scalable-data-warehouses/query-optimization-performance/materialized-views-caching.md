---
course: "building-scalable-data-warehouses"
chapter: "query-optimization-performance"
lesson: "materialized-views-caching"
sourceId: 8039
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-4-query-optimization-performance/materialized-views-caching"
title: "物化视图与缓存层"
description: "利用预计算视图和缓存以减少计算延迟。"
order: 4
plots: ["plots/8039-0.json"]
sourceHash: "eb56fc48afedfc08f76aab1ea19ac7426b5c23b68e30c4e1e5b71f132fb40337"
sourceCorrections: []
---

分布式系统中的优化往往达到一个限度，在此之后，重写逻辑或分区裁剪的效果会越来越小。当特定的聚合模式或复杂的连接逻辑频繁出现时，最有效的方法是通过预先计算结果来完全避免重复计算。性能方程 $T_{总} \approx \frac{V_{扫描}}{R_{磁盘}} + \frac{V_{混洗}}{R_{网络}} + \frac{I_{cpu}}{P_{节点}}$ 说明了物化视图和缓存机制如何针对重复工作负载，将 $V_{扫描}$ 和 $I_{cpu}$ 变量尽可能地降低。

### 持久化层架构

现代MPP（大规模并行处理）平台采用多层架构来提供数据服务。了解数据相对于计算资源的位置对于预测查询延迟是必要的。当用户提交查询时，系统不会立即扫描远程对象存储（如S3或Google Cloud Storage）。相反，它会检查多个易失性和非易失性层。

优先级通常遵循以下顺序：

1. **元数据缓存：** 存储微分区统计信息（最小值/最大值、空值计数）。这使得优化器可以在不列出远程存储文件的情况下编译执行计划。
2. **结果缓存：** 存储之前执行查询的准确输出。这是最快的检索方式，但需要确定性查询匹配。
3. **本地磁盘（SSD）缓存：** 计算节点（虚拟仓库）维护它们最近访问过的原始数据块（微分区）的本地缓存。
4. **远程存储：** 集中式、长期存储层。访问此层由于网络I/O会产生最高的延迟。

> 解耦存储架构中的数据检索层次结构。系统尝试在尽可能高的层满足请求，以减少网络延迟。

### 物化视图

与标准视图不同，标准视图是虚拟的并在运行时执行其定义的SQL，物化视图（MV）将预计算的结果集作为物理实体存储。在MPP系统中，物化视图非常有效，因为它们通常得到透明的维护和使用。

数据库引擎管理物化视图的一致性。当数据载入到基表时，后台服务会更新视图。这在存储成本（用于保存物化数据）和计算成本（用于运行复杂聚合）之间产生一个权衡。

#### 透明查询重写

高级数据仓库的一个突出功能是优化器自动重写查询的能力。用户或BI工具可能查询原始的、细粒度的事实表。然而，如果优化器发现存在满足查询（或其子集）的有效物化视图，它会有效地将执行计划重定向为从预聚合视图读取，而不是从大型基表读取。

这种行为确保下游应用程序无需更改代码即可从性能提升中获益。优化器保证从物化视图提供的数据与基表的当前状态保持一致。

#### 物化成本效益分析

实现物化视图既是技术决定，也是经济决定。您需要支付视图的存储费用以及基表更改时刷新它所需的无服务器计算资源费用。

维护物化视图的成本函数可表示为：

$Cost_{物化视图} = C_{存储} + \sum_{t=0}^{n} (V_{变化} \times C_{处理})$

其中 $V_{变化}$ 是传入的变化数据量。这必须与即席查询的成本进行比较：

$Cost_{即席查询} = N_{查询} \times (V_{扫描} \times C_{计算})$

如果基表频繁更改但很少被查询，维护成本将超过节省的成本。物化适用于需要大量聚合、高基数数据集，这些数据集被频繁读取但更新频率适中。



![成本盈亏平衡点：物化视图 vs. 即席查询](plots/8039-0.json)



> 比较成本分析，显示盈亏平衡点。物化视图产生较高的初始固定成本（维护和存储），但每次查询的边际成本显著较低。

### 实现与局限

在定义物化视图时，工程师必须遵循特定的确定性限制。大多数MPP平台不支持物化视图定义中的非确定性函数（如`CURRENT_TIME`或`RANDOM`）或复杂的自连接，因为这些会使增量维护在计算上不可行。

考虑一个场景，有一个PB级的 `SALES` 表，仪表板需要按区域划分的每日收入。

```sql
-- 高效物化视图定义
CREATE MATERIALIZED VIEW mv_daily_sales_region
CLUSTER BY (region_id, sale_date)
AS
SELECT 
    region_id,
    sale_date,
    SUM(amount) as total_revenue,
    COUNT(transaction_id) as transaction_count
FROM raw_sales_data
GROUP BY region_id, sale_date;
```

在此示例中，聚簇键应用于视图本身。即使视图小于基表，优化其存储布局对于获得最佳性能仍然是必要的。

### 利用结果缓存

结果缓存位于服务层，是最积极的优化形式。如果用户执行的查询与最近执行的查询（由具有相同角色访问权限的任何用户）完全匹配，系统会立即返回存储的结果，而无需启动计算集群。

要触发结果缓存，通常需要满足以下条件：

1. **语法或语义匹配：** SQL文本必须相同，或者解析后的逻辑计划必须匹配。
2. **基础数据稳定性：** 自生成缓存结果以来，属于基表的微分区不得发生更改。
3. **上下文 (context)一致性：** 影响输出的会话参数 (parameter)（如时区设置）必须匹配。

这个缓存层是短暂的，通常持续24小时。它对于那些经常使用相同的“刷新”查询轮询数据库的仪表板工具很有用。为了最大化结果缓存命中，工程师应标准化BI工具中的SQL生成，以确保查询签名保持一致。

### 本地磁盘缓存（数据缓存）

结果缓存避免了计算，而本地磁盘缓存则加速了计算。当虚拟仓库（计算集群）启动时，它最初是“冷的”，意味着它必须从远程对象存储中拉取所有所需数据。随着查询的运行，仓库将检索到的微分区存储在本地SSD上。

后续需要相同数据分区的查询将从SSD而不是网络读取。这就创建了一个“热”集群状态。性能工程通常涉及围绕保持集群运行以维护此缓存（防止自动挂起）与闲置计算资源成本之间的策略。

对于关键的报告SLA，通常的做法是在仓库恢复后立即运行一个预备脚本来扫描相关表段，以“预热缓存”。这确保了第一个查询系统的人工分析师能够体验到SSD级别的延迟，而不是网络级别的延迟。

## 参考资料

- [Database Management Systems](https://books.google.com/books?id=QJzT3K6jU_UC) — Raghu Ramakrishnan, Johannes Gehrke (2003)
  Publisher: McGraw-Hill Education
  全面概述数据库系统，包括查询优化和物化视图的基本概念。
- [The Snowflake Elastic Data Warehouse](http://www.vldb.org/pvldb/vol10/p1779-dageville.pdf) — Benoit Dageville, Allison Lee, Tharindu Dananjaya, Shreya Prasad, Greg Rahn, Kenton Murray, Shishir Kumar, Karthik Raman, Steven Lee and Ashish Motivala (2017)
  Journal: Proceedings of the VLDB Endowment (PVLDB); Publisher: VLDB Endowment; Volume: 10; Pages: 1779-1790; DOI: [10.14778/3137628.3137637](https://doi.org/10.14778/3137628.3137637)
  描述了领先的MPP数据仓库Snowflake的架构，详细介绍了其解耦的存储和计算以及多层缓存机制（结果缓存、本地SSD缓存、远程存储），与课程内容直接相关。
- [Using Materialized Views](https://docs.snowflake.com/en/user-guide/views-materialized) — Snowflake Documentation (2024)
  Publisher: Snowflake
  官方文档，提供了在现代云数据仓库中定义、管理和利用物化视图的实用指南，包括透明查询重写和成本考量。
