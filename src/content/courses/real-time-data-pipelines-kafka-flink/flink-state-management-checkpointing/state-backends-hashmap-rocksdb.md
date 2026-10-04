---
course: "real-time-data-pipelines-kafka-flink"
chapter: "flink-state-management-checkpointing"
lesson: "state-backends-hashmap-rocksdb"
sourceId: 8101
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-3-flink-state-management-checkpointing/state-backends-hashmap-rocksdb"
title: "状态后端：HashMap 对比 RocksDB"
description: "对比堆内 HashMap 状态后端与堆外 RocksDB，以适应不同工作负载需求。"
order: 1
plots: ["plots/8101-0.json"]
sourceHash: "304e473b0b56a6a2535ae956e03842e6a0f4df3fd3db8263c5c941b9f81320c5"
sourceCorrections: []
---

选择状态后端是 Flink 生产部署时一项非常关键的配置决定。它不仅决定了数据存储的位置（Java 堆内存或磁盘），还从根本上影响流式管道的性能特征、垃圾回收行为以及恢复机制。

Flink 提供两种主要的生产可用状态后端：`HashMapStateBackend` 和 `EmbeddedRocksDBStateBackend`。尽管 DataStream API 的抽象使得编写代码时无需担忧具体的存储细节，但底层物理执行在这两种选项之间存在显著差异。

### HashMapStateBackend

`HashMapStateBackend` 将数据作为对象保存在 Java 堆内存中。当应用程序更新 `ValueState<Integer>` 时，Flink 会直接在内存中更新一个 `Integer` 对象。这种方式有效利用了嵌套的 `HashMap` 结构，其中外部 Map 以 KeyGroup 为键，内部 Map 以用户定义的键为键。

由于状态作为原生 Java 对象存在，访问速度非常快。在处理过程中，读取或写入状态无需序列化或反序列化 (SerDe)。该操作只是一个指针解引用，提供 $O(1)$ 的访问效率，开销可以忽略不计。

然而，这种高性能伴随着严格的限制条件。由于状态存在于 JVM 堆上，状态的总大小加上处理所需的内存必须适应配置的堆大小。

#### 内存布局和垃圾回收

在高吞吐量 (throughput)场景下，`HashMapStateBackend` 对垃圾回收器 (GC) 造成较大压力。随着状态对象的创建、修改和丢弃（例如，在窗口操作中），堆的 Old Generation 可能被填满，从而引发完整的 GC 暂停。这些“停顿一切”的事件会暂停处理，导致延迟急剧增加，可能违反严格的服务等级协议 (SLA)。

下图显示了 TaskManager 中使用 HashMap 后端与 RocksDB 后端时的内存布局。

> 将状态存储为堆对象与堆外序列化字节之间的架构差异，会影响延迟和可扩展性。

### EmbeddedRocksDBStateBackend

对于需要海量状态（达到数百 GB 或 TB 级别）的应用程序，依赖 Java 堆是不可行的。`EmbeddedRocksDBStateBackend` 通过将状态存储在本地运行的 RocksDB 实例中来应对此问题。RocksDB 是一种日志结构合并树 (LSM 树) 键值存储，其设计目标是优化快速写入。

使用此后端时，状态存储在堆外原生内存中，当内存缓冲区满时，会溢写到本地磁盘（理想情况下是 SSD）。这种架构将状态大小与 JVM 堆大小解耦。您可以支持仅受集群节点可用磁盘空间限制的状态大小。

#### 序列化开销

这种可扩展性的代价是序列化成本。与 HashMap 后端不同，RocksDB 中的每次读取或写入操作都需要数据跨越 JNI (Java 原生接口) 边界。

1. **写入：** Java 对象 $\rightarrow$ 序列化字节数组 $\rightarrow$ JNI $\rightarrow$ RocksDB 内存表。
2. **读取：** RocksDB $\rightarrow$ JNI $\rightarrow$ 字节数组 $\rightarrow$ 反序列化 Java 对象。

这种序列化开销会引入延迟。虽然通常在毫秒级以下，但对于每秒执行数百万次状态访问的管道来说，它会累积起来。状态访问操作的总成本 $C_{总和}$ 可以近似表示为：

$C_{总和} = C_{计算} + C_{序列化} + C_{JNI} + C_{磁盘I/O}$

其中 $C_{磁盘I/O}$ 仅当请求的数据未在 RocksDB 块缓存中找到且必须从磁盘获取时才变为非零。

### 性能对比分析

在这两种后端之间进行选择，需要分析延迟要求和状态数据量之间的关系。

- **HashMap：** 提供最低的延迟，但当状态大小接近堆限制时会遭遇“断崖式”下降，导致 `OutOfMemoryError` 或大规模 GC 抖动。
- **RocksDB：** 无论状态大小如何，都能提供可预测、一致的性能，前提是本地磁盘 I/O 吞吐量 (throughput)足够。它避免了状态本身的 Java GC 压力，因为数据存在于原生内存中。

下面的图表展示了两种后端的状态大小与处理延迟之间的关系。



![延迟与状态大小概况](plots/8101-0.json)



> HashMap 后端在垃圾回收饱和之前能保持低延迟，而 RocksDB 则随着状态增长保持稳定。

### 快照策略和恢复

后端选择还会影响检查点（应用程序状态的快照）的创建方式。检查点是 Flink 用于保证容错的机制。

**HashMap 快照：**
在过去，此后端需要一种写时复制机制，这在快照阶段可能会占用大量内存。尽管 Flink 为 HashMap 后端提供了异步快照功能，但状态必须在创建检查点时进行序列化。如果状态很大，将整个堆序列化到分布式文件系统（如 S3 或 HDFS）会占用大量网络带宽和 CPU。

**RocksDB 增量检查点：**
RocksDB 后端的一个明显好处是它支持**增量检查点**。由于 RocksDB 不可变数据存储在磁盘上的 SSTables（Sorted String Tables）中，Flink 可以跟踪自上次检查点以来哪些文件已更改。在快照期间，Flink 只上传新的或已修改的 SSTable 文件到检查点存储，而不是整个状态。

此功能对于具有大状态（例如 5TB）但每日“流失”（更改的状态）可能只有 50GB 的管道非常重要。增量检查点显著缩短了检查点过程的持续时间，并降低了维护历史记录相关的存储成本。

### 选择标准

在设计管道时，请根据以下标准选择合适的后端。

1. **状态大小：** 如果每个槽位的预估状态小于 10 GB，`HashMapStateBackend` 通常是安全的。如果每个槽位超过 20-30 GB，则需要 `EmbeddedRocksDBStateBackend` 来避免 GC 问题。
2. **延迟敏感性：** 如果您需要低于毫秒的延迟来进行复杂计算，并且状态能够适应内存，请使用 HashMap。RocksDB 会带来序列化开销。
3. **交互模式：** 如果您的逻辑频繁更新相同的键（热键），HashMap 性能更佳，因为它避免了重复序列化。
4. **可用性：** 如果您需要为大状态进行快速检查点以满足严格的恢复点目标 (RPO)，带有增量检查点的 RocksDB 是更好的选择。

要在应用程序代码中显式配置后端，请使用 `StreamExecutionEnvironment`。然而，通常的做法是在 `flink-conf.yaml` 文件中定义此项，以将基础设施决策与应用程序逻辑分离。

```java
// 以编程方式设置状态后端示例 (不建议用于生产环境的灵活性)
env.setStateBackend(new EmbeddedRocksDBStateBackend(true)); // true 启用增量检查点
```

在下一节中，我们将分析异步屏障快照的运行方式，以便了解 Flink 如何协调这些状态后端来获得一致的分布式状态，而无需暂停数据流。

## 参考资料

- [State Backends](https://flink.apache.org/docs/current/docs/ops/state/state_backends/) — Apache Flink Community (2024)
  Publisher: Apache Software Foundation
  官方文档，详细介绍了 Flink 的状态后端选项、配置以及 HashMap 和 RocksDB 后端的运维考量。
- [RocksDB GitHub Wiki](https://github.com/facebook/rocksdb/wiki) — The RocksDB Project Contributors (2023)
  RocksDB 的官方技术文档，阐述了其架构、LSM 树存储引擎以及以及其处理写入、读取和持久化的方式。
- [Apache Flink: Stream Processing for Real-Time Applications](https://www.manning.com/books/apache-flink-stream-processing-for-real-time-applications) — Fabian Hueske and Vasia Kalavri (2019)
  Publisher: Manning Publications
  一本全面解释 Flink 状态管理、检查点和恢复的图书，为状态后端选择提供了有价值的背景信息。
