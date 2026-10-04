---
course: "intro-data-lake-architectures"
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-5-querying-and-performance"
sourceId: 1398
chapter: "querying-and-performance"
title: "查询与性能"
order: 5
description: "数据湖上Trino和Spark等分布式SQL引擎的性能调优。"
hasQuiz: false
---

确定存储布局和数据摄入管道后，架构需要支持高效的数据检索。将数据存储为Parquet或Avro等优化格式为架构提供了根本支撑，但分析工作负载的性能很大程度上受查询引擎与对象存储层交互方式的影响。与引擎控制物理磁盘的紧耦合数据库不同，数据湖要求查询规划器处理网络延迟和分布式文件系统。

本节考察了Trino和Apache Spark等分布式SQL引擎在解耦存储上执行高性能查询所使用的机制。我们分析了大规模并行处理（MPP）架构如何将SQL语句分解为任务，并分发到工作节点集群。

您将学习最小化I/O操作的技术策略，I/O操作是云分析中的主要成本驱动因素。我们讨论分区修剪和文件跳过，其中引擎利用元数据统计信息、最小值、最大值和空值计数，以完全忽略不相关文件。例如，如果没有修剪，查询操作的时间复杂度与总数据量成正比，即$O(n)$。通过应用元数据过滤器，引擎将扫描限制在相关分区，大幅减小了输入数据量。

本课程还包括：

*   **向量化执行：** 以列式批处理方式处理数据，以高效利用现代CPU指令集。
*   **缓存分层：** 实现SSD或内存缓存，以加速重复访问模式。
*   **查询计划分析：** 读取`EXPLAIN`输出，以诊断瓶颈并验证分区方案是否按预期运行。
