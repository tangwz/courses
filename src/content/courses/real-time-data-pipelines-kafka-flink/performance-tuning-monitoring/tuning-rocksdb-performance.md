---
course: "real-time-data-pipelines-kafka-flink"
chapter: "performance-tuning-monitoring"
lesson: "tuning-rocksdb-performance"
sourceId: 8133
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-7-performance-tuning-monitoring/tuning-rocksdb-performance"
title: "优化 RocksDB 性能"
description: "优化 RocksDB 配置参数以实现高吞吐量的状态访问。"
order: 3
plots: ["plots/8133-0.json"]
sourceHash: "f946752579460edd9c51962b5feddbf96c9484c5296a0d655a6eda9a7ff8b2e7"
sourceCorrections: []
---

管理超出 JVM 堆容量的状态需要转向基于磁盘的嵌入 (embedding)式数据库。Flink 使用 RocksDB 作为这种持久化状态存储，通过在本地文件系统上组织数据来处理数 TB 的状态。与堆状态后端（其中对象以 Java 引用形式存在）不同，RocksDB 要求所有状态对象在存储前序列化为字节数组，并在获取时反序列化。这种架构差异会形成独特的性能特点，其中 CPU（用于序列化）和磁盘 I/O（用于状态访问）成为主要限制因素。

### 日志结构合并树架构

要优化 RocksDB，需要了解它如何在磁盘上组织数据。RocksDB 使用日志结构合并树 (LSM Tree) 数据结构。当 Flink 算子更新状态（例如，`valueState.update()`）时，数据首先被序列化并写入一个内存中的结构，称为 **MemTable**。此操作很快，因为它发生在内存中，并且不会立即触发磁盘 I/O。

一旦 MemTable 达到配置容量，它就变为不可变，并作为排序字符串表 (SSTable) 文件刷新到磁盘。这些文件被组织成不同级别（L0、L1、L2 等）。刷新过程是异步的，但如果 MemTable 填充速度快于后台线程将其刷新到磁盘的速度，RocksDB 会启动“写入停顿”，人为地降低输入速度以防止内存溢出错误。

> 数据从 Flink 算子通过序列化流入 RocksDB LSM 树结构。

### 内存管理和块缓存

Flink 管理 RocksDB 的内存使用，使其与 JVM 堆和谐共存。默认情况下，Flink 将 RocksDB 配置为使用固定量的“托管内存”。这主要分配给 **块缓存**（用于读取）和 **写入缓冲区**（用于 MemTables）。

在高吞吐量 (throughput)场景中，默认分配通常偏向写入缓冲区以支持大量数据写入。但是，如果您的作业大量依赖点查找（例如，将流与大型状态表连接），一个小的块缓存会迫使 RocksDB 频繁从 OS 页面缓存或磁盘获取数据，从而增加延迟。

为了调整这一点，您可以修改写入缓冲区和块缓存之间的关系。RocksDB 的内存消耗大致计算如下：

$M_{\text{总计}} = M_{\text{块\_缓存}} + N_{\text{列}} \times M_{\text{写入\_缓冲区}} \times N_{\text{缓冲区}}$

其中 $N_{\text{cols}}$ 是列族数量（在 Flink 中，基本上每个状态描述符一个），$N_{\text{buffers}}$ 是轮转缓冲区数量。如果您的指标中出现高磁盘读取 I/O，您应该增加托管内存中块缓存的比例，或者增加托管内存的总大小。

### 合并策略

随着 SSTable 在磁盘上积累，RocksDB 必须将它们合并以丢弃被覆盖或已删除的数据（此过程称为合并）。合并可以减少空间使用并提高读取性能，但会消耗 CPU 和磁盘 I/O。Flink 允许您根据工作负载特点选择合并方式。

**分层合并（默认）**
在分层合并中，系统积极地合并文件，以确保每个级别（L1、L2 等）包含不重叠的范围。这最大程度地减少了读取放大，因为读取操作涉及的文件更少。但是，它会产生高写入放大，数据在级别下移时会被多次重写。这非常适合读取密集型工作负载或磁盘空间受限的情况。

**通用合并**
通用合并（类似于 Apache Cassandra 的分层合并）会延迟合并。它刷新 SSTable 并使其大致按时间顺序排列。这显著降低了写入开销，从而实现更高的写入吞吐量 (throughput)。权衡是更高的读取延迟（读取放大），因为查找可能需要检查许多重叠的 SSTable。

对于处理海量数据、状态更新频繁但读取稀疏（例如，仅聚合计数器）的管道，通用合并通常是更好的选择。



![合并策略对 I/O 的影响](plots/8133-0.json)



> 分层合并和通用合并策略在不同工作负载模式下的写入放大比较。

### 使用布隆过滤器优化查找

当 Flink 算子请求一个键时，RocksDB 会首先检查 MemTable，然后是块缓存。如果数据在这两者中都不存在，它必须在磁盘上搜索 SSTable。为了避免扫描每个级别中的每个文件，RocksDB 使用布隆过滤器。

布隆过滤器是一种概率数据结构，它可以告诉你一个键是否*肯定不在*文件中，或者*可能在*文件中。如果过滤器返回否定结果，RocksDB 会完全跳过该文件，从而节省昂贵的磁盘读取。

在 Flink 中，您可以为您的状态描述符启用布隆过滤器。调整参数 (parameter)是每个键的位数。默认通常是 10 位，这会导致大约 1% 的误报率。

$P_{\text{误报}} \approx (1 - e^{-kn/m})^k$

其中 $m$ 是数组中的位数，$n$ 是元素数量，$k$ 是哈希函数数量。增加每个元素的位数可以减少误报（减少不必要的磁盘读取），但会增加过滤器本身的内存开销，该过滤器位于块缓存中。

### 线程和并行度

RocksDB 与 Flink TaskManager 运行在同一进程中，但使用自己的后台线程进行刷新和合并。默认情况下，Flink 可能会为 RocksDB 配置保守数量的线程（通常是 1 或 2 个）。在配备 NVMe SSD 的现代多核实例上，这可能导致应用程序停顿，因为单个刷新线程无法跟上写入速率。

增加 `state.backend.rocksdb.thread.num` 允许 RocksDB 执行并行合并和刷新。但是，设置过高可能会导致上下文 (context)切换开销，并与 Flink 任务线程争夺 CPU 周期。对于高吞吐量 (throughput)节点，建议的起始点是 4 个后台线程，确保至少两个用于刷新（高优先级），以防止 MemTable 阻塞更新。

### 监控性能瓶颈

有效的调整需要精确的遥测数据。您应该通过 Flink 指标组监控特定的 RocksDB 指标：

1. **`rocksdb.estimate-num-keys`**: 追踪您的状态随时间的变化。
2. **`rocksdb.background-errors`**: 此处非零值表示关键 I/O 故障。
3. **`rocksdb.mem-table-flush-pending`**: 如果此值持续很高，则表示您的写入缓冲区配置过小或磁盘速度过慢。
4. **`rocksdb.block-cache-hit` vs `rocksdb.block-cache-miss`**: 用于计算缓存命中率。低于 80-90% 的比率通常表明需要增加托管内存分配。

通过将 RocksDB 配置与流式管道的特定读/写模式对齐 (alignment)，您可以减少延迟峰值，并获得生产级应用所需的稳定吞吐量 (throughput)。

## 参考资料

- [Apache Flink Documentation: The RocksDBStateBackend](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/state/state_backends/#the-rocksdbstatebackend) — The Apache Flink Community (2024)
  Apache Flink中RocksDB配置和调优的官方指南，包括内存管理、线程和监控指标。
- [RocksDB Wiki](https://github.com/facebook/rocksdb/wiki) — The RocksDB Authors (2023)
  RocksDB内部架构、压缩策略（分级、通用）、块缓存和布隆过滤器的权威来源。
- [Database Internals: A Look Inside Distributed Storage Systems](https://www.oreilly.com/library/view/database-internals/9781492040347/) — Alex Petrov (2019)
  Publisher: O'Reilly Media; Pages: 370
  全面解释存储引擎内部原理，包括LSM树、压缩策略和与RocksDB相关的缓存机制。
