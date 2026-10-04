---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink"
sourceId: 256
title: "使用Apache Kafka和Flink的实时数据管道"
description: "利用Kafka和Flink为实时AI应用架构并优化弹性数据流。"
category: "Data Engineering"
level: 3
duration: 35
order: 107
prerequisites: "Kafka、Flink、Java/Scala基础"
color: "cyan"
chapterCount: 8
lessonCount: 40
outcomes: [{"topic": "架构", "description": "设计能够统一批处理和流处理的容错型Kappa架构。"}, {"topic": "可靠性", "description": "使用Kafka事务和Flink两阶段提交，实现精确一次处理语义。"}, {"topic": "状态管理", "description": "配置并调优RocksDB状态后端，以支持流处理应用中的大规模状态管理。"}, {"topic": "优化", "description": "诊断反压、优化序列化并调优并行度，以适应高吞吐量环境。"}]
hasProject: false
---

构建弹性、低延迟数据管道，需要熟练掌握消息代理与流处理引擎间的配合。本课程讲解使用Apache Kafka和Apache Flink搭建生产级流处理系统所需的高级架构模式和实现细节。您将分析分布式系统一致性、状态管理策略以及适用于高吞吐量环境的性能调优方法。

课程内容逐步讲解Kafka的事务协议和Flink检查点算法的内部工作原理。您将实现精确一次处理语义，使用RocksDB管理大规模状态，并应对复杂事件处理场景。本课程内容会涉及实时AI中的特定难题，例如在线特征工程和流式模型服务。课程结束时，您将具备架构、部署和优化数据流的技术能力，以毫秒级延迟为机器学习模型和分析仪表盘提供服务。
