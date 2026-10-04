---
course: "intro-data-lake-architectures"
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-4-metadata-and-cataloging"
sourceId: 1397
chapter: "metadata-and-cataloging"
title: "元数据与编目"
order: 4
description: "管理数据湖元数据。了解 Hive Metastore、AWS Glue 和分区发现。"
hasQuiz: false
---

到架构的这个阶段，你已经配置了对象存储并建立了数据摄取管道。你的数据位于存储桶中，很可能以 Parquet 或 Avro 格式存在。然而，分布式查询引擎无法在没有特定指令的情况下，对这些原始文件执行 SQL 查询。它需要一个模式定义和一个文件位置目录，才能将分散的对象视为结构化表。

本章说明了元数据与编目层，它充当物理存储和计算引擎之间的连接。你将学习 Hive Metastore 和 AWS Glue Data Catalog 如何维护数据湖的状态。我们将演示这些系统如何将逻辑表（例如 `sales_data`）映射到 `s3://bucket/silver/sales/` 等物理位置。

本课程包含以下技术组成部分：

*   **Metastore 抽象：** 逻辑模式如何与物理文件路径解耦。
*   **分区管理：** 用于分区发现的机制，以确保新数据目录一经到位即可立即查询。
*   **治理与安全：** 在编目层面而非文件层面实施基于角色的访问控制 (RBAC)。
*   **血缘追踪：** 审计数据从原始来源流向精选聚合数据的方法。

你将通过设置数据编目并配置爬虫从现有数据集生成表定义，在实践练习中运用这些原理。
