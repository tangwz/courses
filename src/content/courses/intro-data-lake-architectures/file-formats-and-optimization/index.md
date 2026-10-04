---
course: "intro-data-lake-architectures"
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-2-file-formats-and-optimization"
sourceId: 1395
chapter: "file-formats-and-optimization"
title: "文件格式与优化"
order: 2
description: "针对高性能数据湖，对列式格式、压缩和开放表格式进行技术分析。"
hasQuiz: false
---

数据湖建立在对象存储之上，数据以不可变文件而非可变数据库行的形式存在。尽管计算与存储的架构分离提供了可扩展性，但这些文件的物理布局决定了查询延迟和成本。如果查询引擎必须解析数TB的文本日志才能计算一个简单的平均值，系统就会变得低效。本章主要介绍优化这些读取模式的存储层技术细节。

我们首先区分行式格式（例如CSV和Avro）与列式格式（如Apache Parquet和ORC）。你将了解到为什么分析工作负载（通常对大量数据集中的特定指标进行聚合）在数据按列而非按行存储时表现更佳。本课程介绍Apache Parquet的内部结构，详细说明了Run-Length Encoding (RLE)和字典压缩等编码方案如何减少存储占用。

接下来介绍开放表格式。标准文件集合缺少传统数据库中的事务保证。我们将学习Apache Iceberg、Delta Lake和Apache Hudi等规范如何引入元数据层来支持ACID事务和快照隔离。你还将学习如何选择合适的压缩算法，平衡解压的CPU成本与文件较小带来的网络节约。

最后，我们讨论分区策略。一种高效的分区方案允许查询引擎仅处理相关数据子集，这一过程被称为分区修剪。例如，如果目录结构反映了这些键，一个过滤条件为$region = 'US'$和$date > '2023-01-01'$的查询可以跳过扫描湖中大部分数据。学完本章，你将能够配置文件布局，以最大限度地减少I/O并提高吞吐量。
