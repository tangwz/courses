---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-3-flink-state-management-checkpointing"
sourceId: 1416
chapter: "flink-state-management-checkpointing"
title: "Flink 状态管理与检查点"
order: 3
description: "精通 Flink 状态后端。学习配置 RocksDB，管理增量检查点，并处理状态演变。"
hasQuiz: false
---

独立处理单个事件很简单。然而，复杂的流式应用需要上下文。无论是计算移动平均值，还是联接用户活动流，系统都必须保留之前事件的信息。这些保留的信息就是状态。在分布式集群中管理这些数据比较困难，因为即使在硬件故障或网络分区时，系统也必须保证正确性。

本章介绍 Flink 实现容错和一致状态访问的机制。您将分析可用的状态后端，比较在 Java 堆上存储对象与使用嵌入式 RocksDB 键值存储的性能特点。后端选择会影响性能和稳定性，尤其当总状态大小 $S$ 超过可用内存时，通常需要将数据溢出到磁盘。

我们还会讲解 Flink 如何使用异步屏障快照算法来协调分布式快照，以确保精确一次处理。您将实践增量检查点等优化技术，以减少写入持久化存储的数据量。最后，本内容阐述了当应用逻辑变化时，演进状态模式的操作步骤，确保更新不会破坏现有数据。
