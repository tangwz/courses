---
course: "real-time-data-pipelines-kafka-flink"
chapter: "low-level-operations-processfunctions"
lesson: "async-io-external-lookups"
sourceId: 8124
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-5-low-level-operations-processfunctions/async-io-external-lookups"
title: "外部查找的异步 I/O"
description: "使用异步 I/O 丰富流数据与外部数据以保持高吞吐量。"
order: 4
plots: ["plots/8124-0.json"]
sourceHash: "a81ed75980dfe2918116d43398f4cce3ebd6bd16eba7d9a7cdca592c81686776"
sourceCorrections: []
---

使用外部系统存储的数据丰富流事件是许多生产流水线的基本要求。对于变化缓慢、基数较低的参考数据，例如广播状态模式，当丰富数据超出内存限制或更新过于频繁时，这些方法将变得不切实际。例子包括从 NoSQL 数据库获取用户交易历史，或从特征存储中检索预计算的嵌入 (embedding)数据。

使用 `MapFunction` 或 `ProcessFunction` 等标准转换操作符实现这些查找会引入显著的性能瓶颈。这些操作符是同步执行的。当一个操作符向外部数据库发出请求时，处理线程会阻塞，直到响应返回。在高吞吐量 (throughput)流处理系统中，即使是单位毫秒范围内的网络延迟，也会显著降低流水线的整体处理能力。

### 延迟对吞吐量 (throughput)的影响

为了理解异步操作的必要性，请考虑同步系统中延迟与吞吐量之间的数学关系。如果一次外部服务调用需要 $d$ 毫秒才能完成，那么单个处理线程在数学上被限制在特定的吞吐量 $R$。

$R = \frac{1000}{d} \text{ 每秒事件数}$

对于延迟为 10 毫秒 ($d=10$) 的数据库调用，单个线程每秒最多可以处理 100 个事件。要达到每秒 100,000 个事件的吞吐量，Flink 作业将需要 1,000 个并行任务。这会导致资源利用效率低下、高的上下文 (context)切换开销，并增加检查点机制的负载。

异步 I/O 将事件的处理与 I/O 请求的等待时间解耦。Flink 操作符不阻塞线程，而是注册一个回调并立即处理流中的下一个事件。这种并发性使得单个并行单元能够同时处理许多进行中的请求。

> 阻塞式同步执行与非阻塞式异步执行的比较，其中多个请求在时间上重叠。

### 异步 I/O API

Flink 通过 `AsyncDataStream` 工具类和 `RichAsyncFunction` 接口提供了此功能。与立即返回值的标准函数不同，`RichAsyncFunction` 返回 void 并接受一个 `ResultFuture`。实现遵循以下模式：

1. **初始化客户端**：数据库客户端在 `open()` 方法中初始化。此客户端必须支持异步非阻塞操作（例如 Vert.x、不常见的 Java NIO 客户端或标准驱动的异步包装器）。
2. **发送请求**：在 `asyncInvoke(input, resultFuture)` 方法中，应用程序发出查询。
3. **处理回调**：一个回调被注册到数据库客户端返回的 Future 上。当数据库响应时，回调使用丰富后的数据完成 `ResultFuture`。

如果外部系统支持批处理，可以通过在客户端内缓冲请求并按组发送来进一步优化实现，但这会增加超时逻辑的复杂程度。

### 排序模式和一致性

当有多个请求正在处理时，响应可能乱序到达。一个在 $t_1$ 时发送的请求可能在 $t_2$ 时发送的请求之后返回。Flink 提供了两种处理此行为的模式，通过 `AsyncDataStream` 静态方法进行配置。

#### 有序等待

`AsyncDataStream.orderedWait` 确保输出流的顺序严格匹配输入流的顺序。Flink 维护一个内部缓冲区。如果事件 $E_2$ 的结果在事件 $E_1$ 之前到达，Flink 会将 $E_2$ 保留在缓冲区中，直到 $E_1$ 完成并发出。

此模式会引入额外的延迟，该延迟等于当前请求窗口中最慢和最快响应之间的差值。当其下游逻辑依赖精确的事件顺序时，例如在有状态模式匹配中，此模式是必需的。

#### 无序等待

`AsyncDataStream.unorderedWait` 一旦结果可用就立即发出。这最小化了延迟和开销，但改变了流的顺序。然而，在事件时间模式下，这种乱序并非绝对。Flink 确保水位线不会超过记录。

在无序模式下，水位线作为同步屏障。水位线 $W_1$ 和 $W_2$ 之间的事件可能会相互重新排序，但在 $W_1$ 通过后，不会发出 $W_1$ 之前的任何事件。这保持了下游窗口聚合的完整性。

### 管理吞吐量 (throughput)和反压

异步 I/O 操作符维护一个容量参数 (parameter)，用于限制并发进行中请求的数量。这可以防止操作符使外部系统过载或用待处理的 Future 耗尽 Flink 堆内存。

当进行中请求的数量达到此容量时，操作符会触发反压。它会停止从输入流消费，直到至少一个待处理请求完成。此机制向上游传播，自然地减慢源的速度以匹配外部系统的吞吐量能力。



![吞吐量与延迟](plots/8124-0.json)



> 对数比较显示了异步 I/O 如何在延迟增加的情况下保持高吞吐量，而同步处理则迅速下降。

### 超时处理和容错

外部系统容易出现超时和瞬时故障。`asyncInvoke` 方法必须可靠地处理这些情况，以防止数据丢失或无限期停滞。

`AsyncDataStream` 配置接受一个超时参数 (parameter)。如果结果 Future 未在此持续时间内完成，Flink 会调用 `AsyncFunction` 上的 `timeout()` 方法。默认实现会抛出异常并重启作业，这对于轻微的网络瞬断通常是不理想的。

为了实现弹性，请重写 `timeout()` 方法。常见策略有：

1. **重试**：重新发出异步请求（并递减重试计数器）。
2. **侧输出**：将失败的记录发出到侧输出，以便后续检查或放入死信队列。
3. **尽力而为**：如果数据丰富是可选的，则发出部分结果或默认值。

检查点通过等待所有进行中的请求完成，才最终确定快照，从而与异步 I/O 交互。这保证了精确一次语义，但意味着一个挂起的请求会延迟检查点。因此，在数据库客户端本身上配置更严格的超时是更好的选择，而不是仅仅依赖 Flink 的超时机制。

### 实现的最佳做法

1. **执行器池**：不要为数据库回调使用标准 `ForkJoinPool` 或同步执行器。回调逻辑通常在客户端的 I/O 线程上运行。保持回调内部的逻辑尽可能少（例如，仅完成 Future），以避免阻塞 I/O 选择器线程。
2. **使用缓存**：虽然异步 I/O 解决了吞吐量 (throughput)问题，但它不能减少数据库的负载。将异步 I/O 与 `RichAsyncFunction` 内部的内存 Guava 缓存或 Caffeine 缓存结合使用。首先同步检查本地缓存；如果未命中，则继续进行异步调用。
3. **容量调优**：将容量设置得足够高，以覆盖延迟-吞吐量乘积。如果目标吞吐量为 50,000 req/sec 且延迟为 10 毫秒，则容量必须至少为 $50,000 \times 0.01 = 500$ 个请求。设置过低会人为地限制吞吐量；设置过高则存在内存不足（OutOfMemory）错误的风险。

## 参考资料

- [Asynchronous I/O](http://ci.apache.org/projects/flink/flink-docs-release-1.2/dev/stream/asyncio.html) — Apache Flink Documentation (2018)
  Publisher: Apache Flink
  Flink异步I/O API的官方指南，涵盖其用法、排序模式和容错性。
- [Streaming Systems: The What, Where, When, and How of Real-time Data Processing](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) — Tyler Akidau, Slava Chernyak, Reuven Lax (2018)
  Publisher: O'Reilly Media; Pages: 349
  提供流处理系统的基本原理，解释异步操作等功能的动机和影响。
- [Apache Flink: Stream and Batch Data Processing with Dataflow Programming](http://www.vldb.org/pvldb/vol8/p1741-carbone.pdf) — Paris Carbone, Gyula Fekete, Stefan Richter, Sebastian Schelter, Kostas Tzoumas (2015)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 8; Pages: 1741-1752; DOI: [10.14778/2824032.2824076](https://doi.org/10.14778/2824032.2824076)
  一篇基础论文，详细介绍了Flink的架构及其流处理方法，提供了其操作模型的背景知识。
