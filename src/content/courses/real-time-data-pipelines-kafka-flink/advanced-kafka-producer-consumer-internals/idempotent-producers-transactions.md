---
course: "real-time-data-pipelines-kafka-flink"
chapter: "advanced-kafka-producer-consumer-internals"
lesson: "idempotent-producers-transactions"
sourceId: 8093
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-2-advanced-kafka-producer-consumer-internals/idempotent-producers-transactions"
title: "幂等生产者与事务"
description: "实施幂等生产者和事务以确保跨分区的数据完整性。"
order: 2
plots: []
sourceHash: "bc0d3b18c3519b0ba8001c7af3269a680311e317b93f880aa8e94b4ed0dee962"
sourceCorrections: []
---

分布式系统天然需要在一致性和可用性之间做艰难权衡。Apache Kafka的默认配置允许生产者在网络抖动或临时经纪人（broker）故障时重试发送消息。如果原始消息已成功持久化但确认（acknowledgement）在传输中丢失，此重试机制会导致日志中出现重复记录。这种行为满足“至少一次”语义。然而，对于金融账本、库存管理或任何对数据准确性有高要求的有状态聚合，重复数据会引起状态损坏。

为连接可靠传递与数据正确性之间的鸿沟，Kafka引入了两种不同但有关联的功能：幂等生产者和事务API。这些特性将系统从简单的消息传递提升到提供“精确一次”处理语义（EOS）。

### 幂等生产者

幂等性确保特定消息仅被写入特定分区一次，无论生产者发送多少次。这是一项局部保证，作用于单个生产者会话和单个目标分区。它防止网络引起的重复，但不能处理跨分区原子性或应用崩溃。

#### PID和序列号

当`enable.idempotence=true`时，生产者会执行初始化协议。启动时，它会向经纪人请求一个生产者ID（PID）。这个PID对用户是透明的，并在应用重启时重置。

除了PID，生产者为其写入的每个分区维护一个单调递增的序列号（$SeqNum$）。发送给经纪人的每批消息都包含PID和基本序列号。经纪人会在内存和`.snapshot`文件中为每个PID-分区对维护一个已提交的最新序列号（$LastSeqNum$）的映射。

经纪人使用特定逻辑验证传入请求。对于序列号为$S_{new}$的新请求：

1. 如果 $S_{new} = LastSeqNum + 1$，经纪人接受写入并更新 $LastSeqNum$。
2. 如果 $S_{new} \le LastSeqNum$，经纪人将其识别为重试导致的重复。它会立即返回确认，而不再写入数据。
3. 如果 $S_{new} > LastSeqNum + 1$，这表示消息出现间隔，意味着数据流中发生了数据丢失。经纪人会以 `OutOfOrderSequenceException` 拒绝此请求。

这种机制在经纪人层面有效去重消息，性能开销可以忽略不计。它不需要两阶段提交或与其他服务协作。

> 序列号验证逻辑可以防止网络重试时出现重复持久化。

### 原子性多分区写入

幂等性解决了单个流内的重复问题，精密的流水线通常需要在多个组件间具备原子性。一个常见模式是“消费-处理-生产”，即应用从源主题 A 消费，应用业务逻辑，然后将结果写入目标主题 B。

如果应用在写入主题 B 后但在提交主题 A 的偏移量之前崩溃，系统将进入不一致状态。重启后，消费者会再次从主题 A 读取消息（因为偏移量未提交），并向主题 B 生产重复消息。幂等性无法解决此问题，因为新的生产者实例拥有新的PID。

Kafka事务允许应用原子地写入多个分区（并提交消费者偏移量）。要么所有写入都成功并变得可见，要么所有写入都被丢弃。

#### 事务协调器与日志

事务引入了一个名为事务协调器（Transaction Coordinator）的服务器端模块。该组件管理事务的状态。状态持久化在一个名为`__transaction_state`的内部主题中。这个内部主题的运作方式类似于`__consumer_offsets`主题，但使用专门的二进制格式存储事务元数据。

为了使用事务，生产者必须配置一个静态的`transactional.id`。与短暂的PID不同，`transactional.id`在应用重启后依然存在。这种持久性使协调器能够识别同一应用的复活实例。

#### 事务协议

写入路径涉及一个精密的协调协议。

1. **查找协调器：** 生产者查询集群以确定负责其`transactional.id`的事务协调器（通过对ID进行哈希运算确定）。
2. **初始化事务：** 生产者向协调器注册。如果具有相同`transactional.id`的先前实例仍处于活跃状态，协调器会通过递增生产者纪元（epoch）将其隔离。
3. **开始事务：** 生产者开始一个本地事务。
4. **消费与生产：** 生产者向不同分区写入数据。另外，协调器会将“进行中”状态写入事务日志，并记录此事务中涉及的分区。
5. **提交或中止：** 客户端启动一次提交。
6. **两阶段提交 (2PC)：**
   - *准备阶段：* 协调器将`PREPARE_COMMIT`消息写入`__transaction_state`日志。一旦此消息持久化，事务就保证会完成。
   - *提交阶段：* 协调器将特殊的**控制消息**（标记 (token)）写入事务中涉及的每个分区。这些标记对标准消费者不可见，但决定了前面的数据消息是应视为已提交还是已中止。
   - *完成阶段：* 协调器将`COMMITTED`写入事务日志。

> 事务协调器管理生命周期，仅在事务状态持久化后才将提交标记写入数据分区。

### 隔离级别与读取语义

Kafka中事务的实现依赖于生产者立即将数据写入日志，而不论事务状态如何。未提交的消息与已提交的消息一样，都驻留在分区磁盘上。这种区别通过`isolation.level`配置在消费者层面得以实现。

#### read\_uncommitted (默认)

在此模式下，消费者按偏移量顺序读取所有消息。他们将看到来自开放事务的消息，甚至来自已中止事务的消息。这种模式提供较低的延迟，但对事务性工作流没有一致性保证。

#### read\_committed

当消费者配置`isolation.level=read_committed`时，它仅处理截至最新稳定偏移量（LSO）的消息。LSO被定义为最早开放事务的第一个偏移量。

$\text{LSO} = \min(\text{开放事务的起始偏移量}_1, \text{开放事务的起始偏移量}_2, \dots)$

LSO之后出现的消息会在消费者中缓冲，直到事务结果确定。

1. 如果遇到`COMMIT`标记 (token)，缓冲的消息会被发送给应用。
2. 如果遇到`ABORT`标记，缓冲的消息会被丢弃，消费者会跳过此标记。

这种缓冲机制表示，如果并发事务长时间运行，`read_committed`消费者可能会遇到更高的延迟。一个停滞的事务会阻碍LSO的推进，阻止消费者读取同一分区上其他生产者后续的已提交消息。

### 僵尸隔离与纪元

在分布式环境中，一种常见的故障模式是“僵尸生产者”。这发生在生产者进程挂起（例如，在长时间的垃圾回收暂停期间）或失去网络连接时。集群会认为生产者已死，并启动一个新实例。如果旧生产者苏醒并尝试写入，它可能会损坏数据。

Kafka使用纪元隔离来解决此问题。当新的生产者实例使用`transactional.id`初始化时，协调器会为其分配一个更高的纪元号。协调器（以及所有涉及的经纪人）随后将拒绝任何带有旧纪元的写入请求。这种隔离由协议自动处理，确保在任何给定时刻，事务ID只有一个有效的写入者。

### 操作须知

实现事务会产生性能开销。开销来源包括：

- **生产者：** 与协调器的交互以及控制批次的添加。
- **消费者：** `read_committed`模式下的缓冲开销。
- **经纪人：** `__transaction_state`日志的管理和标记 (token)复制。

为优化，生产者应争取更大的事务批次。为每条消息启动和提交事务会因与协调器之间频繁往返而严重降低吞吐量 (throughput)。相反，非常大的事务会增加因LSO延迟而阻塞消费者的风险。一种均衡的做法是根据下游消费者的特定延迟要求调整事务的持续时间和大小。

## 参考资料

- [Apache Kafka Documentation: Producers and Transactions](https://kafka.apache.org/documentation/) — The Apache Software Foundation (2023)
  提供了关于配置和使用Kafka幂等生产者及事务API的全面详细信息，包括其操作影响。
- [KIP-98: Exactly Once Semantics for Message Production](https://cwiki.apache.org/confluence/display/KAFKA/KIP-98+-+Exactly+Once+Delivery+and+Transactional+Messaging) — Jason Gustafson, Flavio Paiva Junqueira, Apurva Mehta, Sriram, Guozhang Wang (2016)
  Journal: Kafka Improvement Proposals; Publisher: Apache Software Foundation
  Kafka幂等生产者的原始设计文档，解释了生产者ID（PID）、序列号和代理验证逻辑。
- [KIP-305: Transactions for Kafka](https://cwiki.apache.org/confluence/display/KAFKA/KIP-305%3A+Transactions+for+Kafka) — Apurva Mehta, Jason Gustafson, Sriram Sundar, Sophie Blee-Goldman (2017)
  Journal: Kafka Improvement Proposals; Publisher: Apache Software Foundation
  Kafka事务API的原始设计文档，解释了事务协调器、两阶段提交和隔离级别。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://learning.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了分布式系统概念、一致性模型和事务处理的基础理解，为Kafka的解决方案提供了更广泛的背景。
