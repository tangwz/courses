---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-2-advanced-kafka-producer-consumer-internals"
sourceId: 1414
chapter: "advanced-kafka-producer-consumer-internals"
title: "高级 Kafka 生产者和消费者工作原理"
order: 2
description: "加深对 Kafka 内部原理的认识。精通复制协议、ISR、分区策略和事务性消息。"
hasQuiz: false
---

基本的生产者配置通常会牺牲持久性以换取吞吐量。当您需要强一致性时，了解分区领导者选举和复制的内部运作方式就变得十分必要。本单元将从默认配置出发，考察 Apache Kafka 如何在分布式环境中保证消息的持久性和顺序。

我们将分析领导者副本与同步副本集 (ISR) 之间的关系。您将看到 `min.insync.replicas` 设置如何与生产者确认机制配合，以防止数据丢失。其中会说明可用性要求，即集群必须保持特定数量的同步节点才能安全地接收写入。

要达到精确一次语义，不仅仅需要重试逻辑。我们将实现幂等生产者和事务性写入，允许应用程序原子地写入多个分区。这保证了一组消息要么完全提交，要么完全丢弃，从而在网络故障期间也能保持数据完整性。

在读取方面，我们将说明消费者组的运作方式。标准的再平衡协议通常会完全暂停消费，从而导致延迟峰值。我们将介绍能够减少这种停机时间的协作式再平衡策略。最后，我们将编写自定义分区器来覆盖默认的分发逻辑，以确保当默认哈希算法不足时，特定键的数据局部性。
