---
course: "real-time-data-pipelines-kafka-flink"
chapter: "low-level-operations-processfunctions"
lesson: "processfunction-hierarchy"
sourceId: 8121
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-5-low-level-operations-processfunctions/processfunction-hierarchy"
title: "ProcessFunction 结构体系"
description: "使用 KeyedProcessFunction 和 CoProcessFunction 访问原始状态和定时器。"
order: 1
plots: []
sourceHash: "269e6c2809694c9076e63e7c3fee23275cb1c4b0a8a54136e761cab4d9328a45"
sourceCorrections: []
---

DataStream API 提供了一种声明式方法来转换流，隐藏了时间与状态管理的复杂性。虽然 `map`、`filter` 和 `window` 等操作符涵盖了大多数用例，但它们通常隐藏了精密事件驱动应用所需的底层机制。当管道需要对状态更新、动态定时器或复杂事件协调进行细致控制时，开发者必须绕过这些抽象，直接使用 `ProcessFunction` 系列。

这个接口展示了 Flink 的基本构成：事件、状态和时间。它作用类似于“扁平映射”操作，并能访问丰富的上下文 (context)对象，从而实现任意流处理逻辑。与独立处理元素的标准转换不同，`ProcessFunction` 结构体系允许与运行时上下文进行交互，从而实现侧输出、定时器注册和直接状态操作等功能。

### 基础：AbstractRichFunction

该结构体系的底层是 `AbstractRichFunction`。虽然它本身不是 `ProcessFunction`，但它作为大多数低层操作的父类。它提供了生命周期方法 `open()` 和 `close()`，这对于初始化不可序列化的资源（如数据库连接或加载静态参考数据）非常重要。

更进一步地，`AbstractRichFunction` 提供对 `RuntimeContext` 的访问。这个上下文 (context)是 Flink 托管状态后端的通道。无论您是使用 `ProcessFunction`、`KeyedProcessFunction` 还是 `CoProcessFunction`，您都可以通过此运行时上下文获得 `ValueState`、`ListState` 或 `MapState` 的句柄。

### KeyedProcessFunction

该结构体系中最常用的变体是 `KeyedProcessFunction`。此函数仅在 DataStream 上进行 `.keyBy()` 操作后才适用。键控流的分区保证使得使用容错的、分区状态成为可能。

在分布式环境中，Flink 根据键对数据流进行分片。`KeyedProcessFunction` 保证所有相同键的事件都由同一个并行任务实例处理。这种局部性允许在处理期间无需网络 I/O 即可高效访问状态后端。

处理元素的方法签名展现了此层级可用的能力：

```java
public abstract void processElement(
    I value,
    Context ctx,
    Collector<O> out
) throws Exception;
```

这里的 `Context` 对象允许访问：

1. **时间戳访问：** 当前元素的事件时间戳。
2. **TimerService：** 注册处理时间或事件时间定时器的机制。
3. **侧输出：** 一种通用方式，用于根据特定条件将数据流拆分为多个流（例如，将格式错误的事件与有效事件分开）。

当定时器触发时，`onTimer` 方法会被调用。此回调使应用能在将来执行操作，例如，如果特定事件未在设定时间间隔内到达，则发出警报。

### CoProcessFunction

实时管道通常需要连接两个不同的流。虽然窗口连接适用于简单的关联，但对于涉及跨多个流的状态机的复杂业务逻辑，它们可能有限制。`CoProcessFunction` 连接两个低层流 `DataStream<IN1>` 和 `DataStream<IN2>`，有效地将它们复用到一个操作符中。

此函数实现了两个处理方法：`processElement1` 和 `processElement2`。由于无法保证两个流之间的到达顺序，应用逻辑必须使用共享状态处理同步。例如，如果流 A 包含“订单已下”事件，流 B 包含“支付已收”事件，则函数可以将首先到达的事件存储在状态中，并等待来自另一个流的对应事件来完成事务。

> 图示了抽象基类与专用处理函数之间的结构关系。

### BroadcastProcessFunction

流处理架构中的一个常见需求是能够在运行时更新处理规则或配置而无需重启任务。`BroadcastProcessFunction` 通过结合一个标准的高吞吐量 (throughput)流与一个低吞吐量控制流来解决此问题。

控制流被“广播”到操作符的所有并行实例。这保证主数据流的每个分区都能访问相同的配置数据。与广播流关联的状态存储在名为 **广播状态** 的 `MapState` 中。与特定键的局部状态（键控状态）不同，广播状态被复制到所有任务。

在实现 `KeyedBroadcastProcessFunction` 时，涉及的逻辑包括：

1. **processElement：** 处理高吞吐量数据流。它可以读取（但不能写入）广播状态以应用当前规则。
2. **processBroadcastElement：** 处理控制流。当新规则到达时更新广播状态。

### 定时器与状态考量

从该结构体系中选择正确的函数主要取决于状态范围和时间需求。

- **键控范围：** 使用 `KeyedProcessFunction`。状态按键隔离。定时器作用于键范围。这是唯一能高效支持事件时间定时器的函数。
- **操作符范围：** 使用 `ProcessFunction`（在非键控流上）。状态绑定到并行任务索引。此功能很少用于业务逻辑，因为它如果管理不当会导致状态分布不均，但它有助于实现自定义源或汇。
- **广播范围：** 使用 `BroadcastProcessFunction`。状态被复制。

这些函数中 `Context` 和 `Collector` 之间的交互是同步的。Flink 保证 `processElement` 和 `onTimer` 对于同一个键永远不会并发调用。这种原子性简化了状态管理，因为开发者无需在函数体内部实现锁或同步块，假设他们修改的是 Flink 管理的状态。

### 处理水位线和时间戳传播

在处理函数结构体系中，时间管理是隐式的，但被严格执行。`Context` 提供对当前水位线的访问，但 `ProcessFunctions` 通常不直接操作水位线。而是对它们做出反应。

当事件触发 `processElement` 时，该事件的时间戳会被分配给任何通过 `Collector` 产生的输出。如果函数在状态中缓存数据并通过定时器稍后发出，则发出事件的时间戳取决于定时器类型。对于处理时间定时器，输出时间戳是系统时钟时间。对于事件时间定时器，输出时间戳对应于定时器注册的时间。

当串联多个处理函数时，理解这种传播很重要。如果 `KeyedProcessFunction` 将数据保留一小时，下游水位线将有效停滞，直到该数据被释放，这可能增加整个管道的延迟。正确平衡状态缓冲与水位线推进是在此抽象级别工作时优化的一个难题。

## 参考资料

- [ProcessFunction](https://flink.apache.org/docs/2.1/docs/dev/datastream/operators/process_function/) — Apache Flink Community (2025)
  Publisher: Apache Flink
  提供了关于 ProcessFunction 及其变体的详细指南，阐述了如何利用状态、定时器和侧输出实现对流处理逻辑的精确控制。
- [State & Fault Tolerance](https://www.bookstack.cn/read/apache-flink-v1.20/docs-concepts-stateful-stream-processing.md) — Apache Flink Community (2024)
  解释了 Flink 有状态流处理的核心原理，涵盖了托管的键控状态、算子状态、广播状态以及确保容错性和一致性的机制，这些是 ProcessFunction 实现的基础。
- [Stream Processing with Apache Flink: Fundamentals, Streamlined Data Pipelines, and Advanced Production Deployments](https://www.oreilly.com/library/view/stream-processing-with/9781491974281/) — Fabian Hueske, Vasiliki Kalavri (2019)
  Publisher: O'Reilly Media
  一本构建 Flink 流处理应用程序的指南，详细介绍了 ProcessFunction 等低级 API 在复杂事件处理、状态管理和高级基于时间的操作中的应用。
- [Event Time](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGl0vfX1AqGkjJ8NcyZwMUToVn-qYS8OUtd-YCvwMAgIOWaaHd-IlYir_xH-kDL4gaB6bq9DjgYT2Xd_soZfqE_kl1iv5q0L1McmqQqbHRtEHrUy9xP-N8Q8SKcZcRcn7pvyajLbx-A8LJLRGuI4ToTDM9HHGR2lKHt5lJEe2ocaf22DhEwDU4kmQ==) — Apache Flink Community (2024)
  Publisher: Apache Flink Community
  阐明了 Flink 的不同时间特性，特别关注事件时间和水位线，这对于理解 ProcessFunction 如何处理定时器并确保事件驱动系统中正确的处理顺序至关重要。
