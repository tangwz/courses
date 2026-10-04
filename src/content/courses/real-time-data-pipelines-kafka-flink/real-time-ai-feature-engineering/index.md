---
course: "real-time-data-pipelines-kafka-flink"
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-8-real-time-ai-feature-engineering"
sourceId: 1423
chapter: "real-time-ai-feature-engineering"
title: "实时AI与特征工程"
order: 8
description: "构建AI管道。计算实时特征，服务模型，并处理流上的请求-响应模式。"
hasQuiz: false
---

机器学习系统只有在输入数据反映环境当前状态时才能有效运行。在传统批处理架构中，特征工程周期性进行，导致在推断时数据过时。本章处理将这些计算转移到流处理环境所带来的工程难题，从而实现模型预测的毫秒级延迟。

我们首先分析在线特征生成。你将编写Flink作业来计算滑动窗口上的聚合，例如计算时间序列的移动平均值。如果模型需要特定特征在窗口 $w$ 上的平均值 $\bar{x}$，流处理器必须维护状态并在新事件到达时增量更新它：

$$ \bar{x}_t = \frac{1}{N} \sum_{i=1}^{N} x_{t-i} $$

在特征生成之后，我们查看模型服务的架构模式。你将比较将模型直接嵌入Flink TaskManager以避免网络开销，与通过HTTP或gRPC使用外部推断服务之间的优劣权衡。本节还包含对资源隔离以及独立于流处理集群进行扩展的讨论。

本章也介绍异步流上的请求-响应模式。由于Kafka将生产者和消费者解耦，实现同步面向用户的应用需要使用临时回复队列的特定关联策略。最后，我们将这些组件与特征存储集成。你将配置管道将计算出的特征写入低延迟存储，保证训练数据集和在线推断环境之间的一致性。
