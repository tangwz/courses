---
course: "building-scalable-data-warehouses"
chapter: "high-throughput-ingestion"
lesson: "practice-cdc-pipeline"
sourceId: 8035
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-3-high-throughput-ingestion/practice-cdc-pipeline"
title: "实践操作：构建CDC管道"
description: "使用开源连接器和目标数据仓库构建可运行的CDC管道。"
order: 5
plots: ["plots/8035-0.json"]
sourceHash: "1f66e6a8649ec9dc383627e3915e2e084bdb383c9ac5329284d08030923b8914"
sourceCorrections: []
---

构建生产级别的变更数据捕获（CDC）管道，需要设计一个能处理事务日志解析、事件序列化和状态重构的系统，这超过了简单复制脚本的能力。一个实际的实现涉及构建一个管道，将数据库变更从事务型源（PostgreSQL）流式传输到分析型目标（如Snowflake或BigQuery）。主要的工程难题包括解码预写日志（WAL）、处理模式差异以及应用幂等合并操作以保持数据一致性。

### 基于日志的管道架构

此架构的基本组成部分是日志提取与日志应用的分离。我们不对源数据库运行 `SELECT *` 查询。相反，我们为数据库的事务日志（即写入磁盘的每项操作的二进制记录）附加一个监听器。

该管道包含三个不同的阶段：

1. **提取：** 连接器读取WAL段并将二进制事件转换为结构化格式（JSON/Avro）。
2. **传输与缓冲：** 消息总线（例如Kafka或Pub/Sub）创建持久缓冲区，使得源和目标可以以不同速度运行。
3. **应用：** 数据仓库消费数据流，将数据存入暂存区，并将其合并到目标表中。

> 数据通过中间事件总线从二进制日志流向分析表。

### 配置逻辑复制

要在PostgreSQL中启用日志提取，我们必须将 `wal_level` 配置为 `logical`。此设置指示数据库记录足够的信息以重构行变更，而不仅仅是用于崩溃恢复的物理磁盘块变更。

您还必须创建一个复制槽。此槽在事务日志中作为游标。它确保数据库引擎不会清除尚未被连接器消费的WAL段。

```sql
-- 验证当前WAL级别
SHOW wal_level;

-- 使用pgoutput插件创建逻辑复制槽
SELECT * FROM pg_create_logical_replication_slot('warehouse_cdc_slot', 'pgoutput');
```

如果消费者（CDC连接器）离线，复制槽将导致WAL文件在源数据库磁盘上累积。这是一个非常要紧的故障模式，需要监控。如果磁盘被占满，源数据库将停止接受写入以防止数据损坏。

### 变更事件的构成

当连接器处理日志条目时，它会发出一个包含数据变更的消息。了解此结构对于后续编写合并逻辑是必需的。一个标准的Debezium风格消息包包含 `before` 和 `after` 状态，以及描述操作类型（`op`）和事务时间戳（`ts_ms`）的元数据。

考虑对 `users` 表进行更新操作。生成的JSON负载结构通常如下所示：

```json
{
  "payload": {
    "before": {
      "id": 101,
      "email": "user@old-domain.com",
      "status": "active"
    },
    "after": {
      "id": 101,
      "email": "user@new-domain.com",
      "status": "active"
    },
    "source": {
      "version": "1.9.5.Final",
      "connector": "postgresql",
      "name": "dbserver1",
      "ts_ms": 1678892345000,
      "lsn": 23405934
    },
    "op": "u",
    "ts_ms": 1678892345123
  }
}
```

在此结构中：

- `op`：表示操作类型（`c` 代表创建，`u` 代表更新，`d` 代表删除）。
- `ts_ms`：连接器处理事件时的纪元时间戳。
- `source.ts_ms`：事务在源数据库中发生时的纪元时间戳。
- `source.lsn`：日志序列号，严格递增，提供顺序保证。

### 暂存与幂等合并

数据仓库摄取层接收这些JSON事件，并将它们追加到原始暂存表。此表是仅追加的事件日志，而不是当前状态的表示。如果同一主键的记录多次变更，它将包含多行。

为了实现最终表，我们应用合并策略。幂等性在此处变得必需。如果管道崩溃并重放最后1000条消息，我们的合并逻辑必须优雅地处理重复数据而不会产生数据异常。

我们使用 `source.ts_ms`（事务时间戳）或 `source.lsn` 来解决冲突。其逻辑是：仅当传入事件比当前存储的更新时，才更新目标表。

#### 合并查询模式

以下SQL模式演示了如何将一批CDC事件从暂存表（`stg_users`）合并到最终表（`dim_users`）。此模式在单个原子事务中处理插入、更新和删除（通过软删除）。

```sql
MERGE INTO dim_users AS target
USING (
    -- 窗口函数，用于为每个主键选择最新的事件
    -- 在当前微批次中
    SELECT *
    FROM (
        SELECT 
            payload:after:id::INT as user_id,
            payload:after:email::STRING as email,
            payload:after:status::STRING as status,
            payload:op::STRING as op_code,
            payload:source:ts_ms::INT as event_ts,
            ROW_NUMBER() OVER (
                PARTITION BY payload:after:id 
                ORDER BY payload:source:ts_ms DESC
            ) as rn
        FROM raw_cdc_staging
        WHERE ingested_at > :last_watermark
    ) 
    WHERE rn = 1
) AS source
ON target.user_id = source.user_id
WHEN MATCHED AND source.op_code = 'd' THEN
    -- 软删除逻辑
    UPDATE SET target.is_deleted = TRUE, target.updated_at = source.event_ts
WHEN MATCHED AND source.event_ts > target.updated_at THEN
    -- 幂等更新：仅在源数据较新时才更新
    UPDATE SET 
        target.email = source.email,
        target.status = source.status,
        target.updated_at = source.event_ts,
        target.is_deleted = FALSE
WHEN NOT MATCHED AND source.op_code != 'd' THEN
    INSERT (user_id, email, status, updated_at, is_deleted)
    VALUES (source.user_id, source.email, source.status, source.event_ts, FALSE);
```

此查询执行几项非常要紧的功能：

1. **批内去重：** `ROW_NUMBER()` 窗口函数确保，如果用户在一小时内更新了五次个人资料，我们只尝试合并该小时的最终状态。
2. **顺序执行：** `ORDER BY payload:source:ts_ms DESC` 保证我们遵循源数据库的时间顺序。
3. **删除处理：** 我们检查 `op` 代码。如果它是 `d`（删除），我们会在数据仓库中执行逻辑删除，而不是物理删除行，从而保留审计历史。

### 延迟与批次效率

构建此管道时，您必须调整缓冲区大小和刷新频率。批次大小（在单个 `MERGE` 语句中处理的CDC事件数量）与端到端延迟之间存在非线性关系。

小批次提供较低延迟，但由于分布式查询引擎的设置和拆解成本，会产生高额开销。大批次对于吞吐量 (throughput)来说是高效的，但会引入延迟。



![摄取延迟与批次大小](plots/8035-0.json)



> 批次大小与处理延迟之间的关系。请注意，在处理量增加持续时间之前，存在一个开销最小化的“最佳点”。

在上图中，观察拐点。在批次大小非常小（左侧）时，延迟主要受连接开销和查询编译时间影响。随着批次大小增加，效率提升，直到数据处理量本身开始线性增加持续时间。对于大多数高吞吐量MPP系统，1到5分钟（或5万到10万行）的批次间隔通常能在准实时报告方面实现最佳平衡。

### 处理模式漂移

CDC管道对模式变更很敏感。如果源PostgreSQL表中添加了列，二进制日志将立即开始包含此新列的数据。如果您的暂存表或 `MERGE` 逻辑未预料到这一点，管道可能会失败。

为了缓解此问题，我们对暂存层使用“读取时模式”方法。通过将CDC负载存储在半结构化列中（例如Snowflake中的 `VARIANT` 或BigQuery中的 `JSON`），当模式演变时，摄取作业不会失败。故障被推送到下游转换层（`dbt` 或SQL视图），这比修复损坏的摄取连接器更容易修复和重放。

在视图定义中使用 `payload:after:new_column` 语法，即使目标模式定义滞后于源系统变更，也允许管道继续运行。这种分离对于保持数据平台操作的高可用性非常要紧。

## 参考资料

- [Kafka: The Definitive Guide: Real-Time Data and Stream Processing at Scale](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043089/) — Gwen Shapira, Todd Palino, Rajini Sivaram, Krit Petty (2021)
  Publisher: O'Reilly Media
  Apache Kafka 的综合指南，解释了其架构、流处理以及作为高吞吐量数据摄取管道消息总线的作用，对传输层至关重要。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley-vch.de/en/books/the-data-warehouse-toolkit-the-definitive-guide-to-dimensional-modeling-978-1-118-53080-1) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  数据仓库方法的奠基性著作，提供了 ETL 策略、数据集成和在分析系统中维护数据一致性的背景，与 CDC 目标相关。
- [Snowflake Documentation: MERGE Statement and Working with Semi-Structured Data](https://docs.snowflake.com/en/sql-reference/sql/merge) — Snowflake Inc. (2024)
  Publisher: Snowflake Inc.
  Snowflake 官方文档，详细介绍了用于幂等更新、插入和软删除的 `MERGE` SQL 语句，以及在数据仓库中使用 `VARIANT` 等半结构化数据类型处理模式演变的策略。
