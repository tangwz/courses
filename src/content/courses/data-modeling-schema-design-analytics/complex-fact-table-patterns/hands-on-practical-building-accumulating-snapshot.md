---
course: "data-modeling-schema-design-analytics"
chapter: "complex-fact-table-patterns"
lesson: "hands-on-practical-building-accumulating-snapshot"
sourceId: 7989
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-4-complex-fact-table-patterns/hands-on-practical-building-accumulating-snapshot"
title: "动手实践：构建累积快照"
description: "为工作流构建累积快照"
order: 5
plots: ["plots/7989-0.json"]
sourceHash: "d9e08f134588f3b0d5beef26f95f58f0cf06317c65168f1d22ce6d2cfec75fe7"
sourceCorrections: []
---

累积快照事实表解决了一个事务型模式处理不佳的特定分析问题：追踪业务流程的生命周期。事务表将单个事件记录为独立行，而累积快照则为每个业务实体（例如订单、索赔或支持工单）创建一行记录，并随着流程的进展更新该行。

这种结构对分析有明确始末的工作流很有用。它让分析人员能够使用简单的列计算来计算阶段之间的时间间隔（通常称为“滞后”），而非在庞大的事务日志上执行复杂的自连接。

### 规划模式结构

在此实践练习中，我们将构建电商订单履约流程模型。一个订单会经过四个明确的里程碑：

1. 订单已下达
2. 支付已确认
3. 仓库已发货
4. 客户已收货

在标准的事务型模型中，这些将是 `events` 表中的四条不同的记录。在我们的累积快照中，这将是一条包含多个日期列的记录。

该模式需要一个标识流程的主键（订单ID），以及指向相关维度（客户、产品、地域）的外键。其显著特点是代表这些里程碑的日期/时间列集合。

```sql
CREATE TABLE fct_order_processing_snapshot (
    order_key           INT PRIMARY KEY,
    customer_key        INT,
    order_id            VARCHAR(50),
    -- 里程碑时间戳
    order_placed_at     TIMESTAMP,
    payment_confirmed_at TIMESTAMP,
    shipped_at          TIMESTAMP,
    delivered_at        TIMESTAMP,
    -- 度量值
    order_amount        DECIMAL(10,2),
    -- 状态
    current_status      VARCHAR(20),
    last_updated_at     TIMESTAMP
);
```

### 可视化转换过程

从事务型视图到累积快照的转变涉及将事件行透视成列属性。下图说明了“订单 #101”的多条事件记录如何合并成快照表中的一条记录。

> 离散事件行转换为单一完整工作流记录。

### 填充快照表

填充此表需要不同于标准事实表的策略。通常，数据仓库倾向于仅追加操作。然而，累积快照本质上需要更新。当现有订单发生“已发货”事件时，我们必须修改现有行而非插入新行。

在Snowflake、BigQuery或Databricks等现代数据仓库中，`MERGE` 语句是处理此逻辑的标准方式。该逻辑将传入事件数据与现有快照进行比较。如果订单已存在，它会更新与新事件对应的特定时间戳列。如果是一个新订单，则插入新行。

假设有一个源数据流 `stg_order_events` 包含今日变更。我们将其应用于快照表：

```sql
MERGE INTO fct_order_processing_snapshot AS target
USING (
    SELECT 
        order_id,
        customer_key,
        event_type,
        event_timestamp,
        amount
    FROM stg_order_events
) AS source
ON target.order_id = source.order_id
WHEN MATCHED THEN
    UPDATE SET
        target.payment_confirmed_at = CASE 
            WHEN source.event_type = 'PAYMENT_SUCCESS' THEN source.event_timestamp 
            ELSE target.payment_confirmed_at 
        END,
        target.shipped_at = CASE 
            WHEN source.event_type = 'SHIPPED' THEN source.event_timestamp 
            ELSE target.shipped_at 
        END,
        target.delivered_at = CASE 
            WHEN source.event_type = 'DELIVERED' THEN source.event_timestamp 
            ELSE target.delivered_at 
        END,
        target.last_updated_at = CURRENT_TIMESTAMP(),
        target.current_status = source.event_type
WHEN NOT MATCHED AND source.event_type = 'ORDER_PLACED' THEN
    INSERT (
        order_key, customer_key, order_id, 
        order_placed_at, order_amount, 
        current_status, last_updated_at
    )
    VALUES (
        source.order_id, -- 为简化起见，假设为一对一映射
        source.customer_key,
        source.order_id,
        source.event_timestamp,
        source.amount,
        'ORDER_PLACED',
        CURRENT_TIMESTAMP()
    );
```

此查询处理工作流逻辑。如果发货事件到达，查询会找到在“订单已下达”阶段创建的行，并填充 `shipped_at` 列的空值。

### 分析工作流延迟

实施此模式的主要原因是衡量效率。业务负责人经常会问：“一个已支付订单需要多长时间才能发货？” 在事务表中，回答这个问题需要按订单ID分组并计算两种不同事件类型 `MIN(time)` 之间的差值。而在累积快照中，查询只是简单的减法运算。

$Lag_{履约} = T_{发货} - T_{支付}$

在SQL中，这转化为直接的聚合操作：

```sql
SELECT 
    AVG(DATEDIFF(hour, payment_confirmed_at, shipped_at)) as avg_发货小时数,
    AVG(DATEDIFF(hour, shipped_at, delivered_at)) as avg_运输小时数
FROM fct_order_processing_snapshot
WHERE order_placed_at >= '2023-01-01'
  AND delivered_at IS NOT NULL;
```

### 管理完整性和空值

累积快照中的行很少会立即完整。在工作流的早期阶段，下游列（例如 `delivered_at`）将为空。这是预期行为。这些空值表明流程尚未达到该里程碑。

为了监控流程的健康状况，您可以可视化停滞在特定阶段的订单量。以下图表演示了订单如何像漏斗一样，100% 的订单都已下达，但随着它们进入后续阶段，数量会减少或出现滞后。



![订单流程吞吐量](plots/7989-0.json)



> 累积快照中达到各里程碑的订单量。

### 处理乱序事件

实际数据管道可能存在杂乱情况。累积快照必须能够应对迟到数据。例如，由于系统延迟，一个“已发货”事件可能比“支付”事件先到达数据仓库，即使事件实际上是按顺序发生的。

前面提供的 `MERGE` 语句能很自然地处理这种情况。由于我们根据事件类型独立更新列，ETL批次中的到达顺序就不那么重要了。如果“已发货”时间戳首先写入，`payment_confirmed_at` 列将保持为空，直到支付记录到达。一旦两条记录都处理完毕，该行就完整了，无论其摄取顺序如何。

但是，如果重新处理历史数据，您必须确保更新逻辑不会用旧数据覆盖有效数据。一个常见的保障措施是添加一个条件来检查传入时间戳是否确实比现有值新，但对于列代表不同里程碑的累积快照，这比在类型1维度中的风险要小。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://books.google.com/books/about/The_Data_Warehouse_Toolkit.html?id=z1Y8AgAAQBAJ) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons
  这本数据仓库和维度建模的奠基性著作，提供了各种事实表类型的全面指南，包括累积快照事实表的详细解释及其分析应用。
- [MERGE](https://docs.snowflake.com/en/sql-reference/sql/merge) — Snowflake Inc. (2024)
  Publisher: Snowflake Inc.
  SQL MERGE 语句的官方文档，对于在 Snowflake 等现代数据仓库环境中高效更新和插入数据到累积快照事实表至关重要。
