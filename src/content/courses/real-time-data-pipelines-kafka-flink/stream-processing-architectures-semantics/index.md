---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-1-stream-processing-architectures-semantics"
sourceId: 1412
chapter: "stream-processing-architectures-semantics"
title: "流处理架构与语义"
order: 1
description: "分析流处理模式。了解 Lambda 与 Kappa 架构以及分布式系统中的数据一致性语义。"
hasQuiz: false
---

构建有弹性的流式应用程序，需要清晰地了解数据如何在分布式环境中流动和持久化。与数据集有限的批处理不同，流处理处理的是持续到达且通常乱序的无界数据集。本部分确立了使用 Apache Kafka 和 Flink 设计生产级数据管道所需的架构原则。

我们将从审视分布式日志开始。这种只追加的数据结构构成了 Kafka 存储引擎的底层，并实现了生产者与消费者之间的解耦。你将分析日志如何支持顺序访问模式和数据重放以应对系统故障。

在存储层之后，我们将对比两种架构模式：Lambda 和 Kappa。Lambda 架构为批处理和流处理维护独立层，而 Kappa 架构通过将所有数据视为流来简化技术栈。我们将阐述 Flink 的一致性机制如何使 Kappa 架构能够替代复杂的 Lambda 部署。

本章以处理保证和时间的专业定义作结。你将辨别至多一次、至少一次和精确一次的传输语义。此外，我们还将通过区分事件时间 $t_e$ 和处理时间 $t_p$ 来处理时间偏差问题。理解这些定义是正确处理延迟数据并确保应用程序中结果确定性所必需的。
