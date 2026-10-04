---
course: "real-time-data-pipelines-kafka-flink"
chapter: "advanced-kafka-producer-consumer-internals"
lesson: "replication-protocols-isrs"
sourceId: 8091
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-2-advanced-kafka-producer-consumer-internals/replication-protocols-isrs"
title: "复制协议和ISR"
description: "分析 Kafka 复制机制、领导者选举和同步副本 (ISR) 要求。"
order: 1
plots: ["plots/8091-0.json"]
sourceHash: "5daf45d48b7ee82ef1502055047f642469b4db36c85ef9e9b5afef3ae146cb24"
sourceCorrections: []
---

分布式系统建立在硬件终会发生故障的假设上。单节点数据库中的磁盘故障常常导致数据丢失。为了减少这种风险，Apache Kafka 依赖分区复制来确保数据持久性。然而，简单地将数据复制到多个节点对于高吞吐量 (throughput)环境来说是不够的。了解数据如何从领导者传播到其追随者的具体机制，以及集群如何确定消息何时被安全地“提交”，这些都非常重要。

### 领导者-追随者架构

Kafka 中的每个分区都有一个副本被指定为**领导者**。所有生产和消费请求（默认情况下）都通过这个领导者进行。其余的副本是**追随者**。

Kafka 中的追随者在集群协调方面是被动的，但在数据获取方面是主动的。它们本质上作为专门的消费者运行。追随者向领导者发送 `FetchRequest` 消息，请求从特定偏移量开始的数据。领导者返回数据，追随者将其添加到其本地日志中。这种“拉取”模式可以防止领导者在追随者无法跟上摄入速率时使其过载。

这种交互对于性能非常重要。领导者不会推送数据；它会等待追随者请求数据。这种机制使得 Kafka 能够处理具有不同网络能力或磁盘 I/O 速度的追随者，而不会阻塞领导者，前提是持久性设置不严格强制同步复制。

### 同步副本的定义

并非所有副本都相同。Kafka 维护一个动态副本集合，称为**同步副本 (ISR)**。此列表包括领导者和任何在功能上“已赶上”领导者的追随者。

历史上，Kafka 使用消息计数滞后和时间滞后来定义“已赶上”。现代版本几乎完全依赖时间。如果追随者在 `replica.lag.time.max.ms` 定义的时间窗内向领导者发送了 `FetchRequest`，则它被视为同步的。

如果追随者未能发出获取请求，或者它发出了请求但在此时间窗内无法赶上领导者的日志末端偏移量 (LEO)，则领导者会将该追随者从 ISR 中移除。该追随者仍然是一个副本，但它实际上是“不同步的”。

### 日志末端偏移量和高水位线

为了弄清 Kafka 如何保证一致性，我们必须区分为每个分区副本维护的两个指针：

1. **日志末端偏移量 (LEO)：** 写入日志的最后一条消息的偏移量。
2. **高水位线 (HW)：** 成功复制到当前 ISR 中所有副本的最后一条消息的偏移量。

高水位线充当消费者的屏障。消费者只能读取到高水位线之前的消息。高水位线和领导者 LEO 之间的消息被视为“未提交”，并且不暴露给消费者。这可以防止“幻读”问题，即消费者读取的消息因领导者故障而丢失。

高水位线由领导者计算。它有效地跟踪 ISR 集合中的最小 LEO。

$HW = \min(\{LEO_{r} \mid r \in ISR\})$

当生产者发送一条消息时，它会被追加到领导者的日志中（增加领导者的 LEO）。追随者获取此数据并更新自己的 LEO。一旦领导者收到获取请求，表明所有 ISR 成员都已拥有该消息，领导者就会推进高水位线。

> 领导者跟踪所有追随者的复制状态。在此状态下，高水位线为 100，因为追随者 2 尚未确认偏移量 101。如果追随者 3 的滞后超过时间阈值，则它可能位于 ISR 之外。

### 持久性与吞吐量 (throughput)

持久性保证的强度由生产者的 `acks` 配置和代理的 `min.insync.replicas` 设置决定。

当生产者设置 `acks=all`（或 `-1`）时，领导者会等到消息写入 ISR 的**所有**当前成员的本地日志后才发送确认。这提供了最强的持久性保证。如果领导者崩溃，ISR 中的任何特定追随者都有资格成为新的领导者，而不会丢失数据。

然而，如果 ISR 缩小到只剩下领导者，仅 `acks=all` 并不能保证数据安全。如果所有追随者都滞后或崩溃，ISR 集合的大小将变为 1。领导者在将数据写入自己的磁盘后会立即确认写入。如果该单个领导者发生故障，数据就会丢失。

为了防止这种情况，您必须在代理或主题级别配置 `min.insync.replicas`。此设置充当看门人。

$\text{成功写入} \iff |ISR| \ge \text{min.insync.replicas}$

如果可用同步副本的数量低于此阈值，领导者会拒绝 `acks=all` 的生产请求，并抛出 `NotEnoughReplicasException` 异常。这有效地停止了管道以保持数据一致性。

### 可用性权衡

此配置引入了可用性与一致性之间的经典分布式系统权衡。

- **高可用性：** `min.insync.replicas = 1`。只要至少有一个节点（领导者）存活，集群就接受写入。风险：如果领导者在复制完成前发生故障，数据可能会丢失。
- **高一致性：** `min.insync.replicas = 2`（在复制因子为 3 的设置中）。集群要求至少有两个节点确认写入。风险：如果两个节点发生故障（或进行维护），分区将变为只读（不可用于写入）。

以下图表展示了复制延迟和生产者吞吐量 (throughput)与所需确认数量之间的关系。



![确认级别对写入延迟的影响](plots/8091-0.json)



> 随着吞吐量的增加，要求所有副本同步（`acks=all`）的延迟成本相比仅领导者确认的方式显著增加。

### 非干净领导者选举

当 ISR 中的所有副本都发生故障时，就会出现灾难性情况。集群只剩下 ISR 之外的追随者（滞后）或根本没有在线副本。

如果一个滞后的追随者最终重新上线，Kafka 会遇到一个由 `unclean.leader.election.enable` 配置管理的困境：

1. **False (默认值)：** 系统会等待原始 ISR 的成员重新上线。这优先考虑一致性。分区将保持不可用，直到前 ISR 成员恢复。
2. **True：** 系统会选举第一个可用副本作为新领导者，即使它不在 ISR 中。这会立即恢复可用性，但会保证数据丢失，因为新领导者缺少在旧领导者上已提交的消息。

对于数据完整性是绝对要求的金融或审计追踪管道，此设置必须保持 `false`。对于正常运行时间比丢失少量日志更重要的指标或日志流，`true` 可能是可以接受的。

### 控制复制流量

复制流量会与生产者和消费者流量竞争网络带宽。在高性能调优中，如果您的硬件允许，您可能会将复制流量隔离到单独的网络接口，尽管这需要高级代理配置。

更常见的是，您会调整获取大小。`replica.fetch.max.bytes` 设置控制追随者在单个请求中尝试获取多少数据。增加此值可以提高高容量主题的吞吐量 (throughput)，但会增加代理的内存压力，如果网络是瓶颈，还可能导致延迟峰值。

知晓这些内部协议，您可以构建出无需人工干预即可承受节点故障的系统，同时准确预测一致性要求会带来多少延迟。

## 参考资料

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  官方文档提供了Kafka复制协议、ISR、LEO、HW以及acks和min.insync.replicas等配置的权威详细信息。
- [Kafka: The Definitive Guide, 2nd Edition](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043048/) — Gwen Shapira, Todd Palino, Rajini Sivaram, Krit Petty (2021)
  Publisher: O'Reilly Media
  一本全面的书籍，涵盖Kafka的架构，包括对复制、ISR和一致性保证的详细解释。
