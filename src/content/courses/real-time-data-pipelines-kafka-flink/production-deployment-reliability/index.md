---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability"
sourceId: 1421
chapter: "production-deployment-reliability"
title: "生产部署与可靠性"
order: 6
description: "确保生产可靠性。处理序列化、schema 演进，并集成 Kafka Connect。"
hasQuiz: false
---

将 Flink 应用从本地开发环境迁移到分布式生产集群，需要高度关注数据完整性和运行稳定性。在单机上正常运行的代码，在分布式系统固有的可变延迟和部分故障面前，经常会失效。本章讨论了维护长期运行数据管道所需的工程标准。

我们首先分析序列化效率。像 JSON 这样的文本格式会带来显著的存储和处理开销。你将使用 Apache Avro 和 Protocol Buffers 实现二进制序列化，以减小有效载荷大小。为了管理数据结构随时间的变化，我们引入了 Schema Registry，它在解耦的生产者和消费者之间强制执行兼容性约定。这确保了对生产者 schema 的更新不会破坏下游的消费者应用。

接着，重点转向集成和弹性。我们使用 Kafka Connect 作为一种标准化的机制，用于在 broker 之间传输数据，减少对自定义连接器代码的需求。最后，我们定义了用于处理“毒丸”消息的恢复策略——这些消息是导致处理失败的畸形数据事件——并概述了在基础设施中断时避免数据丢失的措施。
