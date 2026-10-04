---
course: "intro-data-lake-architectures"
chapter: "architectural-foundations"
lesson: "medallion-architecture"
sourceId: 7997
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-1-architectural-foundations/medallion-architecture"
title: "奖章架构"
description: "将数据湖结构化为青铜（原始）、白银（清洗）和黄金（整理）层，以提高数据质量。"
order: 3
plots: []
sourceHash: "9cd59d38de046bc76dac869c39b3493fd61494e6d38f3a27e444a2b9a9c937bd"
sourceCorrections: []
---

数据湖以低成本提供几乎无限的存储容量，但这种灵活性带来了较大的组织管理难题。若无严密的管理结构，数据湖会迅速恶化为难以管理的零散文件集合，常被称为“数据沼泽”。为保持数据质量和可靠性，工程师们采用奖章架构。这种设计模式根据数据的质量和验证级别，将数据分成青铜、白银和黄金三个明确的层次。

这种架构的主要目的是逐步提高数据质量。随着数据流经这些层次，它变得更干净、更有条理、更聚合。这种多步骤方法让不同角色（数据工程师、数据科学家和业务分析师）能够在最适合其特定需求的阶段获取数据。

### 青铜层

青铜层，常被称为“原始区”或“着陆区”，作为所有数据进入数据湖的最初入口。此层次的优先事项是写入速度和历史数据保留，而非读取性能或数据清洁度。

青铜层中的数据通常是只追加、不可更改的源系统记录。它保留源数据的原始格式，例如 JSON、CSV、XML 或外部数据库转储。工程师不应在摄取过程中更改数据内容。如果源系统在应为整数的字段中发送字符串，青铜层将存储该字符串。这种忠实记录确保您始终拥有源系统生成内容的完整记录。

**青铜层数据的核心特点包括：**

- **原始真实性：** 保持数据接收时的精确状态。
- **元数据列：** 包括额外的技术列，如 `_ingestion_timestamp` 和 `_source_file_name`，以追踪数据来源。
- **批次组织：** 通常按对象存储分区组织，依据数据加载的日期和时间（$year=YYYY/month=MM/day=DD$）。

此层次作为一个安全网。如果下游转换逻辑中出现错误，您可以修复代码并从青铜层重新运行转换管道，无需再次向源系统请求数据。

### 白银层

白银层表示数据的验证和丰富版本。青铜层是原始数据的堆放，而白银层是可信资产。在此层次中，数据经过筛选、清洗和补充。它通常采用高性能的列式存储格式，如 Apache Parquet，并常通过事务日志（如 Delta Lake 或 Apache Iceberg）管理，以有效处理更新和删除。

从青铜层到白银层的转换涉及“清洗”操作。您强制执行模式、处理空值、移除重复记录并转换数据类型。例如，青铜层中的时间戳字符串在白银层中变为真实的时间戳对象。

白银层在数据湖中充当企业数据仓库视图。它通常是规范化的（第三范式），并包含原子数据而非聚合数据。数据科学家经常查询白银层，因为它提供训练机器学习 (machine learning)模型所需的干净、细粒度数据，没有业务聚合带来的预计算偏差。

**白银层中的典型转换：**

- **去重：** 删除可能已被多次摄取的重复条目。
- **数据丰富：** 如果对于基本理解很重要，关联参考数据（如通过 `store_id` 查找 `store_name`）。
- **模式强制：** 拒绝与预期结构不匹配的记录，或将它们隔离以供审查。

### 黄金层

黄金层是精心整理、聚合的层次，专为特定业务用例设计。白银层围绕数据主题（客户、产品、订单）组织，而黄金层围绕项目特定需求（季度销售报告、客户流失分析）组织。

黄金层中的数据经过高度转换。它将业务逻辑应用于白银层中的细粒度数据。这通常涉及大量聚合，例如汇总每日销售额、计算移动平均值或确定复杂的关键绩效指标。这里的数据模型经常从规范化结构转变为维度模型，例如星型模式，以优化 Tableau、PowerBI 或 Looker 等 BI 工具的读取性能。

因为数据是预计算和聚合的，查询延迟大幅降低。业务分析师和高管可以有效使用这些数据，无需理解原始摄取或清洗逻辑的深层机制。

> 数据在奖章架构中的流转过程。数据从原始摄取（青铜）到验证结构（白银），最后到聚合的业务指标（黄金）。

### 分层的好处

采用这种分层方法提供了结构上的隔离。原始摄取过程（青铜）中的故障不会立即破坏管理层仪表板（黄金），因为黄金层表会保持其上次已知良好状态，直到管道恢复。

此外，这种分离符合数据湖“读时模式”的灵活性，同时在上层提供了“写时模式”的可靠性。向青铜层写入数据时限制最少，以快速捕获数据。向白银层和黄金层写入数据时附带明确的限制，以确保下游用户的可靠性。

通过将摄取格式与消费格式解耦，您可以针对不同的限制进行优化：

1. **青铜层：** 优化写入吞吐量 (throughput)和大数据量存储。
2. **白银层：** 优化查询灵活性和数据完整性。
3. **黄金层：** 优化读取延迟和业务逻辑表达。

接下来的章节中，我们将分析这种逻辑架构如何映射到具体的物理实现方案，例如 Lambda 和 Kappa 架构，以处理数据在这些层次间流动的时序。

## 参考资料

- [Medallion Architecture | Databricks Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvVEePvhu5SlDZkrI6O_JWMJT99hxocbvmL0dS1Y16PNaNOlPkXT7OrNLqg2YFsI7T5ZGfQTgv41UYkNPhWCZfvylWYpmMOqazde4oyovaw67tdhUbqOtbrK0WpEFqBXzSxQxV9fpbxjL3IH4HNrGHg_WKA==) — Databricks (2024)
  Publisher: Databricks
  官方文档，详细介绍了奖章架构、其三个层（青铜、白银、黄金）以及在数据湖仓中实施的实践。
- [Delta Lake Documentation](https://delta.io/docs/) — The Delta Lake Project (2025)
  Publisher: The Linux Foundation Projects
  提供Delta Lake的官方信息，这是一个开源存储层，为数据湖带来ACID事务和可靠性，常用于在奖章架构中实现白银层。
- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108269/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  一本关于现代数据工程原理、模式和架构考虑的综合书籍，涵盖与奖章架构相关的数据湖设计和数据质量策略。
