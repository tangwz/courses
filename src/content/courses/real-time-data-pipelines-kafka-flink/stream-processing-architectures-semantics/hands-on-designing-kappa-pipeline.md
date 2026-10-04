---
course: "real-time-data-pipelines-kafka-flink"
chapter: "stream-processing-architectures-semantics"
lesson: "hands-on-designing-kappa-pipeline"
sourceId: 8089
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-1-stream-processing-architectures-semantics/hands-on-designing-kappa-pipeline"
title: "动手实践：设计一个Kappa管道"
description: "动手实践设计一个使用Kafka和Flink的完整Kappa架构管道。"
order: 5
plots: ["plots/8089-0.json"]
sourceHash: "303e1b577620a70a228814ea84487c25790e4602cd3989e7ee1805805b14c0cc"
sourceCorrections: []
---

设计Kappa架构需要您在看待数据保留和处理逻辑的方式上进行根本性转变。传统的Lambda架构维护两个独立的 codebase：一个用于低延迟速度（流处理），另一个用于高准确性完整性（批处理）。Kappa架构将日志确立为单一事实来源。流处理引擎同时处理实时数据摄入和历史数据重处理。

本实践部分介绍了点击流分析管道的设计。目标是实时计算用户会话时长和购物车放弃率，同时保留在业务定义改变时重新计算历史指标的能力。我们将使用Apache Kafka作为不可变日志，并使用Apache Flink进行统一计算来构建此管道。

### 不可变日志配置

此架构的基础是Kafka主题的保留策略。在Kappa设计中，原始输入主题必须作为规范数据集。与数据消费后即被删除的瞬态消息队列不同，此主题有效地替代了数据湖，用于原始事件存储。

对于我们的点击流主题 `raw-clickstream-events`，如果标准保留策略基于大小或时间，且在数据可能需要重处理之前就过期，则这些策略不够用。您必须使用Kafka的分层存储或配置由足够磁盘空间支持的长期保留期。

输入主题的配置要求：

- **retention.ms**：设置为 `-1`（无限）或覆盖您合规审计窗口的持续时间（例如，7年）。
- **replication.factor**：最小值为3，确保数据持久性。
- **min.insync.replicas**：设置为2，以在数据摄入时加强一致性。

> Kappa架构中的数据流，显示了实时和重处理作业针对同一个不可变日志的并行执行。

### 模式定义与演进

由于日志是持久的，数据结构必然会演变。需要一个模式注册表来管理兼容性。对于高吞吐量 (throughput)的管道，Protocol Buffers (Protobuf) 或Avro由于紧凑的序列化和严格的类型检查，优先于JSON。

我们定义事件模式以包含 `event_timestamp` 字段。此字段驱动Flink中的事件时间处理，与Kafka的摄入时间戳不同。

```protobuf
syntax = "proto3";

message ClickEvent {
  string user_id = 1;
  string session_id = 2;
  string url = 3;
  string event_type = 4; // 例如，“查看”，“添加到购物车”，“结账”
  int64 event_timestamp = 5; // 纪元毫秒
  map<string, string> properties = 6;
}
```

当演进此模式时，您必须确保向后兼容性。新字段必须是可选的或具有默认值，以便Flink作业可以使用新逻辑（使用Schema V2编译）读取历史数据（使用Schema V1写入）。

### Flink作业结构和时间语义

Flink作业处理无界流。为了支持Kappa的“批处理”能力（重处理），作业必须是确定性的。这在很大程度上依赖于正确的Watermark策略。

当重处理一年的数据时，处理时间 $t_p$ 将进展极快，而事件时间 $t_e$ 跟踪历史时间戳。如果您的窗口逻辑依赖于 $t_p$（例如，`ProcessingTimeSessionWindows`），您的历史结果将与实时结果不同。您必须使用 `EventTimeSessionWindows`。

考虑检测“会话”的逻辑。会话在一段不活动时间后关闭。

$\text{会话间隔} > 30 \text{ 分钟}$

在代码中，我们定义源并显式分配时间戳和Watermark，以处理乱序数据，这在分布式收集中断常见。

```java
DataStream<ClickEvent> stream = env
    .fromSource(kafkaSource, WatermarkStrategy
        .forBoundedOutOfOrderness(Duration.ofMinutes(1))
        .withTimestampAssigner((event, timestamp) -> event.getEventTimestamp()),
        "Kafka 源");

stream
    .keyBy(ClickEvent::getUserId)
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .process(new SessionAnalyticsFunction())
    .sinkTo(databaseSink);
```

### 实现重处理的幂等性

Kappa的决定性特点是改进逻辑并对历史数据进行重跑的能力。假设您部署了具有30分钟会话间隔的管道。随后，数据科学分析表明15分钟的间隔为预测模型提供了更好的特征。

要在没有单独批处理系统的情况下实现此更改：

1. **更新逻辑**：将 `EventTimeSessionWindows.withGap` 参数 (parameter)修改为15分钟。
2. **新消费者组**：为Flink作业配置一个新的Kafka消费者组ID（例如，`analytics-v2`）。
3. **从最早开始**：将起始偏移策略设置为 `Earliest`。

当新作业启动时，它从主题的开头读取。这可能产生一个问题：下游数据库将收到旧记录的更新。

为了处理这个问题，Sink必须是**幂等的**。

如果sink是键值存储或支持更新/插入操作的数据库（如PostgreSQL的 `INSERT ON CONFLICT` 或Elasticsearch的 `_update`），重放将正确覆盖先前的值。如果系统需要仅追加的sink（如写入S3），您必须将重处理后的输出定向到新位置（例如，`s3://bucket/analytics/v2/`），以避免损坏现有数据集。

### 处理重放期间的吞吐量 (throughput)

Kappa架构中的一个常见挑战是“追赶”阶段。当重放数月的数据时，摄入到Flink的速度将比实时流量高出几个数量级。

如果Flink操作符或Sink无法处理这种激增，您可能会观察到背压。为了在设计阶段减轻这种情况：

1. **并行度调整**：确保Kafka主题有足够的（例如64或128个）分区，以便在重放时实现高并行度，即使实时流量只需要并行度为4。
2. **状态后端**：使用RocksDB作为状态后端。在重放过程中，将生成大量状态（开放窗口）。基于堆的后端很可能会遇到内存不足错误。
3. **Sink缓冲**：增加sink配置中数据库提交的批次大小，以优化高吞吐量写入，而不是低延迟可用性。

以下图表说明了重放事件与稳定状态处理期间的资源使用情况。



![资源利用率：稳定状态 vs. 历史重放](plots/8089-0.json)



> 吞吐量比较显示了重放作业处理历史数据所需的巨大峰值，一旦追赶完成或终止，最终降至零。

通过配置日志的无限保留、强制执行严格的模式演进以及在Flink中设计确定性的事件时间逻辑，您消除了对单独批处理层的需求。流处理器成为用于过去和未来数据的统一引擎。

## 参考资料

- [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) — Jay Kreps (2013)
  Publisher: LinkedIn Engineering Blog
  介绍了Kappa架构的概念，并强调日志作为核心数据抽象。
- [Apache Kafka Documentation](https://kafka.apache.org/documentation) — Apache Kafka (2024)
  Publisher: The Apache Software Foundation
  提供Kafka配置、操作实践和架构的官方详细信息。
- [Apache Flink Documentation](https://flink.apache.org/docs/) — The Apache Software Foundation (2025)
  Publisher: The Apache Software Foundation
  阐述了Flink的编程模型、时间语义、状态管理和部署。
- [Kafka: The Definitive Guide: Real-time data and stream processing at scale](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043072/) — Gwen Shapira, Todd Palino, Rajini Sivaram, Krit Petty (2021)
  Publisher: O'Reilly Media; Pages: 488
  全面介绍了Apache Kafka的核心概念、架构和实际应用。(第二版)
- [Stream Processing with Apache Flink: Fundamentals, StreamSQL, and real-time data systems](https://www.oreilly.com/library/view/stream-processing-with/9781491974286/) — Fabian Hueske, Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  涵盖了Apache Flink的流处理能力，包括事件时间、有状态处理和容错性。
