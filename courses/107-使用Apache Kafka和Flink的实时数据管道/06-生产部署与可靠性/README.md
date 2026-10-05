# 第 6 章：生产部署与可靠性

来源：[原章节](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability)

[返回课程目录](../README.md)

将 Flink 应用从本地开发环境迁移到分布式生产集群，需要高度关注数据完整性和运行稳定性。在单机上正常运行的代码，在分布式系统固有的可变延迟和部分故障面前，经常会失效。本章讨论了维护长期运行数据管道所需的工程标准。

我们首先分析序列化效率。像 JSON 这样的文本格式会带来显著的存储和处理开销。你将使用 Apache Avro 和 Protocol Buffers 实现二进制序列化，以减小有效载荷大小。为了管理数据结构随时间的变化，我们引入了 Schema Registry，它在解耦的生产者和消费者之间强制执行兼容性约定。这确保了对生产者 schema 的更新不会破坏下游的消费者应用。

接着，重点转向集成和弹性。我们使用 Kafka Connect 作为一种标准化的机制，用于在 broker 之间传输数据，减少对自定义连接器代码的需求。最后，我们定义了用于处理“毒丸”消息的恢复策略——这些消息是导致处理失败的畸形数据事件——并概述了在基础设施中断时避免数据丢失的措施。

## 小节

- 1. [Avro 和 Protobuf 序列化](01-Avro%20%E5%92%8C%20Protobuf%20%E5%BA%8F%E5%88%97%E5%8C%96.md)
- 2. [模式注册表集成](02-%E6%A8%A1%E5%BC%8F%E6%B3%A8%E5%86%8C%E8%A1%A8%E9%9B%86%E6%88%90.md)
- 3. [Kafka Connect 用于源和目标](03-Kafka%20Connect%20%E7%94%A8%E4%BA%8E%E6%BA%90%E5%92%8C%E7%9B%AE%E6%A0%87.md)
- 4. [故障恢复策略](04-%E6%95%85%E9%9A%9C%E6%81%A2%E5%A4%8D%E7%AD%96%E7%95%A5.md)
- 5. [实战：运行中的模式演进](05-%E5%AE%9E%E6%88%98%EF%BC%9A%E8%BF%90%E8%A1%8C%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%BC%8F%E6%BC%94%E8%BF%9B.md)
