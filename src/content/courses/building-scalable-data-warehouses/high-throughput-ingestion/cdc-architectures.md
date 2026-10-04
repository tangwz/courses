---
course: "building-scalable-data-warehouses"
chapter: "high-throughput-ingestion"
lesson: "cdc-architectures"
sourceId: 8031
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-3-high-throughput-ingestion/cdc-architectures"
title: "变更数据捕获架构"
description: "实现基于日志的CDC，以进行接近实时的数据同步。"
order: 1
plots: ["plots/8031-0.json"]
sourceHash: "ecd7aa170bb68c8b80d758e969a82981c49a0465b2000572d7717954a5849d50"
sourceCorrections: []
---

依赖 `updated_at` 时间戳或递增整数ID的基于查询的复制策略，对现代数据平台造成显著限制。这些方法会产生高延迟时段，增加源数据库查询引擎的负载，而且，无法捕获硬删除。为实现高吞吐量 (throughput)数据摄取和接近实时的一致性，工程师必须采用基于日志的变更数据捕获 (CDC)。这种架构完全绕过SQL层，直接从数据库的预写日志 (WAL) 或二进制日志读取，以提取修改事件流。

### 基于日志的提取基础

关系数据库通过事务日志确保持久性。在更改提交到存储文件 (表空间) 之前，它会添加到序列化日志结构中。在PostgreSQL中，这是预写日志 (WAL)；在MySQL中，是二进制日志 (binlog)；在Oracle中，是重做日志。

基于日志的CDC工具充当复制客户端。它们向数据库注册，以接收这些日志记录的连续流。因为此过程涉及读取二进制文件，而不是执行SQL `SELECT` 语句，所以对源系统的性能影响可以忽略不计。

提取过程将内部数据库日志格式转换为平台无关的事件流。单个事件通常封装行状态的转换。如果我们将时间 $t$ 时行 $r$ 的状态表示为 $r_t$，则CDC事件 $E$ 捕获该转换：

$E_t = (r_{t-1}, r_t, \text{元数据})$

此处，$r_{t-1}$ 表示“之前”的映像 (特定更新逻辑所需)，$r_t$ 是“之后”的映像，元数据包括操作类型 (INSERT, UPDATE, DELETE) 和事务提交时间戳。

> 架构流程显示日志提取与主数据库引擎分离。

### 变更事件的结构

在分布式数据仓库环境中，有效载荷结构决定了合并数据的效率。标准CDC有效载荷包含重建当前状态所需的数据以及正确排序事件所需的元数据。

以Debezium等连接器生成的典型JSON有效载荷结构为例：

1. **有效载荷 (之后):** 行的新状态。
2. **有效载荷 (之前):** 先前状态 (用于识别主键变更或特殊增量计算)。
3. **源元数据:** 包括服务器ID、事务ID和日志序列号 (LSN)。
4. **操作码:** 表示操作的单个字符 (`c` 表示创建，`u` 表示更新，`d` 表示删除，`r` 表示快照读取)。

包含“之前”映像可以进行高级验证。例如，如果数据仓库接收到UPDATE操作，但“之前”映像与数据仓库中的当前状态不匹配，这可能表示事件遗漏或同步问题。

### 处理删除和模式演变

基于日志的CDC相对于轮询的主要优势是可靠地捕获 `DELETE` 操作。当记录从源表中物理删除时，轮询查询将不再返回它。数据仓库无法区分已删除记录与未更改记录。

在基于日志的架构中，删除会触发特定事件类型 (`op: 'd'`)。摄取管道必须通过在目标数据仓库中执行逻辑删除 (软删除) 或物理删除来处理此操作。一种常见模式是将此删除标记 (token)追加到原始历史表，并使用视图将其过滤掉：

$视图_{当前} = \{x \in 表_{历史} \mid x.op \neq 'd' \land x.timestamp = \max(timestamp)\}$

模式演变是另一个显著的复杂情况。当源中添加或重命名列时，二进制日志格式会更改。CDC连接器使用模式注册表。连接器检测日志中的DDL变更，更新注册表中的模式定义，并对后续消息进行版本控制。数据仓库摄取管道必须配置为处理这些演进的模式，通常通过使用半结构化数据类型 (如Snowflake的 `VARIANT` 或BigQuery的 `JSON` 类型) 来适应新字段，而不会中断管道。

### 快照与一致性

纯日志读取器无法从头构建数据库副本，因为日志经常被轮换或归档。为了初始化数据仓库，系统必须执行“一致性快照”。

最有效的快照技术涉及无锁方法：

1. 连接器开启一个可重复读事务。
2. 它记录当前日志位置 (LSN)。
3. 它扫描完整的表，将它们作为 `READ` 事件发出。
4. 扫描完成后，它会切换到从记录的LSN开始流式传输日志。

这种转换确保零数据丢失。但是，这会引入一个时段，在此期间数据仓库可能会收到尚未进行快照的记录的更新。管道逻辑必须处理“upsert”语义，以确保在初始加载记录之前到达的更新不会被旧数据覆盖。

### 资源利用比较

转向CDC会显著改变数据库的资源情况。轮询会产生周期性的CPU峰值 (锯齿状模式)，其扩展性与数据集大小不佳，而CDC保持一个恒定的、低开销的占用，与数据变更速率 (速度) 成比例，而非总数据量。



![资源影响：轮询对比CDC](plots/8031-0.json)



> CPU负载曲线比较。轮询在执行期间会产生高负载峰值，而CDC则保持一个稳定的、较低的基线，这取决于事务量。

### 分布式系统中的架构挑战

实施CDC会带来分布式系统方面的挑战，这些挑战在批处理中不太常见。

**顺序保证：**
数据库日志是严格有序的，但消息总线 (例如：Kafka) 通常仅在分区内保证顺序。如果相同主键的事件分布在不同分区中，数据仓库可能会在相应的INSERT之前处理UPDATE。为避免此情况，生产者必须将主键用作分区键。这确保了特定实体的所有事件都落在同一分区中并按顺序消费。

**重复投递：**
CDC管道通常保证“至少一次”投递。网络超时可能导致连接器重发未确认的事件。摄取层必须幂等。如果数据仓库两次收到相同事件 (相同的LSN/偏移量)，则第二次应用必须导致相同的状态。我们将在关于幂等的下一节中为此给出数学证明。

**事务边界：**
单个数据库事务可能更新十个不同的表。在日志中，这些事件表现为一系列用 `BEGIN` 和 `COMMIT` 标记 (token)包围的事件。大多数CDC连接器将这些作为独立记录进行流式传输。如果数据仓库以微批次方式摄取这些记录，则可能只加载了事务的一半，导致临时的参照不一致。高级实现会缓冲事件，直到观察到提交标记，或者使用流处理框架在写入数据仓库之前重新组装事务。

## 参考资料

- [Debezium Documentation](https://debezium.io/documentation/) — The Debezium Community (2024)
  日志型变更数据捕获的全面指南，涵盖各种数据库连接器、事件结构、快照和模式演变。
- [Chapter 30. The Write-Ahead Log (WAL)](https://www.postgresql.org/docs/current/wal.html) — The PostgreSQL Global Development Group (2025)
  Publisher: The PostgreSQL Global Development Group
  PostgreSQL官方文档，解释了其预写式日志的作用及其在持久性和复制中的角色，这是PostgreSQL日志型CDC的基础。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本涵盖分布式系统基本概念的著作，对高吞吐量数据管道至关重要，包括数据模型、复制、一致性和流处理。
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/) — The Apache Software Foundation (2024)
  Apache Kafka官方文档，提供了消息总线架构、分区排序和与CDC管道相关的交付语义等基本信息。
