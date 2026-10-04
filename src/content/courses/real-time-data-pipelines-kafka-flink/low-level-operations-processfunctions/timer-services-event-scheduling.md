---
course: "real-time-data-pipelines-kafka-flink"
chapter: "low-level-operations-processfunctions"
lesson: "timer-services-event-scheduling"
sourceId: 8122
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-5-low-level-operations-processfunctions/timer-services-event-scheduling"
title: "定时器服务与事件调度"
description: "注册处理时间定时器和事件时间定时器，用于调度流内的未来动作。"
order: 2
plots: ["plots/8122-0.json"]
sourceHash: "b26037f102068e07c1880d8d4524614ed906f9dd63ea4c84a646c14c0807b23a"
sourceCorrections: []
---

有状态流处理常常需要一些逻辑，这些逻辑的处理时间会超出事件到达后的即时处理。虽然窗口算子根据固定的时间边界聚合数据，但许多实际场景需要任意的时间相关行为。例如，一个应用可能需要等待一个互补信号15分钟，或者在一个会话长时间不活动时触发一个警报。这些模式需要对时间进行精确控制，这是通过 Flink `KeyedProcessFunction` 中的 `TimerService` 来实现的。

### TimerService 接口

`TimerService` 是数据处理逻辑与 Flink 内部时间系统之间的桥梁。通过 `processElement` 方法中的 `Context` 对象可以访问它，它允许应用查询当前时间并注册未来的回调。

有必要了解 Flink 中的定时器本质上是限定在某个键上的。当定时器触发时，Flink 会恢复定时器注册时所活跃的键控状态上下文 (context)。这保证了 `onTimer` 回调可以访问与该特定实体相关的正确状态变量（例如，`ValueState`，`ListState`）。因此，定时器仅在 `KeyedProcessFunction` 或应用于 `KeyedStream` 的算子中可用。

`TimerService` 会提供两种不同的当前时间：

1. **当前处理时间**：执行任务的机器的实际时钟时间。
2. **当前水位线**：由上游水位线定义的逻辑事件时间进展。

### 处理时间与事件时间调度

选择处理时间调度还是事件时间调度，会根本上改变流水线的确定性和正确性。

**处理时间定时器** 依赖于处理机器的系统时钟。当调用 `ctx.timerService().registerProcessingTimeTimer(timestamp)` 时，定时器在机器的实际时钟达到指定时间戳时触发。这些定时器对与延迟SLA或外部系统交互相关的超时很有用，例如重试异步数据库调用。然而，它们是非确定性的；重新处理历史数据将导致不同的触发时间，并可能产生不同的结果，因为回放速度与原始数据摄取速度不同。

**事件时间定时器** 运用水位线机制。通过 `ctx.timerService().registerEventTimeTimer(timestamp)` 注册的定时器仅在水位线达到或超过指定时间戳时触发。这保证了确定性。如果作业停止并从一个保存点重放，或者历史数据被重新处理，事件时间定时器将在相对于数据流的精确相同逻辑时刻触发，确保状态转换的一致性。

以下图表展示了水位线进展与定时器执行之间的关系。随着水位线（蓝线）单调增加，它与注册的定时器阈值（红点）相交，触发回调。



![事件时间进展与定时器执行](plots/8122-0.json)



> 水位线进展确定性地触发事件时间定时器，不受处理速度影响。

### 定时器的内部管理

定时器并非临时的内存对象；它们是 Flink 状态的一个组成部分。当定时器注册时，它与 `ValueState` 和 `MapState` 一起持久化在状态后端中。这种持久化保证了容错性。如果 TaskManager 发生故障，已注册的定时器将从最新的检查点恢复，从而避免回调丢失。

Flink 采用优先级队列来高效地管理这些定时器。在 Heap 状态后端中，这是一个基于 Java 堆的优先级队列。在 RocksDB 状态后端中，定时器存储在一个专门的列族中，按时间戳排序。这种架构使得 Flink 可以高效地轮询下一批需要过期的定时器，而无需扫描所有键。

一个需要注意的实现细节是定时器去重。Flink 为每个 `(key, timestamp, namespace)` 元组仅维护一个定时器。如果为同一键多次注册时间戳为 $T$ 的定时器，它将在优先级队列中产生一个条目，并且 `onTimer` 只会被调用一次。这种行为对性能很有利，因为它在处理高频流时，可以防止定时器队列膨胀，即使多个事件试图在同一未来时刻调度超时。

### onTimer 回调

当满足定时器条件时（系统时间超过时间戳或水位线超过时间戳），Flink 会调用 `onTimer` 方法。

$T_{触发} \leq \max(T_{水位线}, T_{系统})$

`onTimer` 方法签名提供了一个 `OnTimerContext`，它允许应用执行以下操作：

1. 访问触发回调的时间戳。
2. 与 `TimeDomain`（处理时间与事件时间）交互。
3. 修改键控状态。
4. 向主输出或侧输出发送元素。

由于 `onTimer` 与 `processElement` 是同步的（它们绝不会对同一个键并发运行），因此开发者在访问状态变量时无需实现锁定机制。

### 优化：定时器合并

在高吞吐量 (throughput)场景中，为每个事件注册一个定时器可能会降低性能。每次注册都涉及序列化开销和向状态后端优先级队列的插入操作。

一种常见的优化策略是**定时器合并**。应用不会为 `current_time + delay` 注册定时器，而是将目标时间四舍五入到更粗的粒度，例如下一秒或下一分钟。

例如，如果一个事件在 $t=10,123$ 到达并需要5秒超时，一个简单的实现会在 $15,123$ 注册一个定时器。如果合并到1秒的粒度，应用会在 $16,000$ 注册定时器。

$T_{注册} = \lceil \frac{T_{事件} + \Delta}{精度} \rceil \times 精度$

这项技术显著减少了 RocksDB 的独立写入操作数量，并减少了 `onTimer` 的调用次数，因为 Flink 有效地批量处理了该秒的回调。

> 合并操作将多个事件触发器对齐 (alignment)到单个时间戳，从而减轻状态后端压力。

### 清理定时器

定时器正确的生命周期管理对于避免状态膨胀是不可或缺的。虽然定时器一旦触发就会自动从状态后端消失，但经常会出现这样一些情况：在挂起定时器执行之前，它就变得无效了。例如，如果设置了一个“会话超时”定时器，但有一个新的用户活动事件到达，那么之前的超时应该有效地取消或忽略。

`TimerService` 提供了 `deleteEventTimeTimer` 和 `deleteProcessingTimeTimer` 方法。然而，在 RocksDB 中删除定时器涉及读-修改-写循环（写入一个墓碑），这可能开销很大。

一种替代的高性能模式是避免显式删除。相反，应用将“有效”时间戳存储在 `ValueState` 中。当 `onTimer` 触发时，代码会将触发时间戳与状态中存储的时间戳进行比较。如果它们不匹配，回调会认为该定时器已被取代，不执行任何操作。这种“惰性”取消通常在写入负载高的系统中比即时删除提供更好的吞吐量 (throughput)。

## 参考资料

- [Streaming Systems: The What, Where, When, and How of Real-Time Data Processing](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) — Tyler Akidau, Slava Chernyak, Reuven Lax (2018)
  Publisher: O'Reilly Media
  解释了流处理的基础概念，包括事件时间、处理时间以及水位线，这些对于理解 Flink 中的时间依赖逻辑至关重要。
- [Apache Flink: Stream and Batch Processing in a Single Engine](https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper5.pdf) — Fabian Hueske, Aljoscha Krettek, Robert Metzger, Till Rohrmann, Stefan Richter, Arne Simon, Kostas Tzoumas, and Stephan Ewen (2015)
  Journal: Proceedings of the 7th Biennial Conference on Innovative Data Systems Research (CIDR '15); Pages: 62-73; DOI: [10.5441/002/cidr15-05](https://doi.org/10.5441/002/cidr15-05)
  介绍了 Flink 架构及其用于有状态流处理和时间管理的核心功能。
- [Stream Processing with Apache Flink: Fundamentals, Streamlining Dataflow, and Advanced Applications](https://learning.oreilly.com/library/view/stream-processing-with/9781491974308/) — Fabian Hueske, Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  详细解释了 Flink 的 ProcessFunction 和 TimerService，并提供了关于状态和时间管理的实用建议。
