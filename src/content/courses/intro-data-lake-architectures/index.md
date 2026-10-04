---
course: "intro-data-lake-architectures"
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures"
sourceId: 250
title: "数据湖架构入门"
description: "采用现代设计模式、优化存储格式和数据摄取管道，构建高性能数据湖。"
category: "Data Engineering"
level: 2
duration: 14
order: 132
prerequisites: "SQL与编程要点"
color: "cyan"
chapterCount: 5
lessonCount: 26
outcomes: [{"topic": "架构模式", "description": "采用Medallion模式（青铜、白银、黄金层）设计多层数据架构。"}, {"topic": "存储格式", "description": "采用Apache Parquet实现列式存储，并借助Iceberg等开放表格式管理事务。"}, {"topic": "数据摄取", "description": "构建批处理和流式管道，将数据可靠地从源系统移入数据湖。"}, {"topic": "查询优化", "description": "采用分区、文件裁剪和分布式查询引擎，优化数据检索速度。"}]
hasProject: false
---

数据湖是现代人工智能和数据分析系统中原始与处理数据的核心存储地。本课程讲解构建可扩展、高性能数据湖所需的技术架构。我们将界定其结构层，从原始数据摄取到适用于机器学习 (machine learning)和报告的精处理表格。课程内容介绍Apache Parquet和Avro等存储格式，Apache Iceberg和Delta Lake等开放表格式，以及计算与存储的分离。您将配置元数据目录，实现数据摄取管道，并运行分布式查询。内容着重于架构模式，特别是Medallion架构，并提供关于分区策略和模式管理的技术指导。
