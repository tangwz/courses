---
course: "building-scalable-data-warehouses"
chapter: "advanced-data-modeling-scale"
lesson: "handling-semi-structured-data"
sourceId: 8028
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale/handling-semi-structured-data"
title: "处理半结构化数据"
description: "原生摄取、展平及查询嵌套JSON/Parquet数据的方法。"
order: 3
plots: ["plots/8028-0.json"]
sourceHash: "acacb313eba49530041b3d4c29e9e852004130647d5fe6956f43c7c469eea676"
sourceCorrections: []
---

现代数据仓库已不再固守数据在摄取前必须严格结构化的要求。应用日志、点击流和面向文档的数据库的兴起，使得JSON、Avro和Parquet等半结构化格式成为数据工程中的主要组成部分。对于大规模并行处理（MPP）架构，处理这些格式需要改变建模策略。您不再需要立即将复杂的层级结构完全规范化为关系型表。相反，您可以使用Snowflake、BigQuery和Redshift等平台的原生功能，高效地存储嵌套数据，同时通过标准SQL接口提供访问。

### 半结构化数据的原生存储

过去，在关系型数据库中存储JSON意味着将文本字符串插入到`VARCHAR`或`CLOB`列中。这种方法效率不高，因为每个查询都需要解析整个字符串来提取单个属性。现代MPP系统对此的处理方式不同。当您将半结构化数据加载到专用数据类型（例如Snowflake中的`VARIANT`、Redshift中的`SUPER`或BigQuery中的`STRUCT`/`ARRAY`）时，存储引擎不仅仅是存储原始文本。

相反，引擎会即时执行一个通常称为“拆解（shredding）”或“列式化（columnarization）”的过程。它识别文档的内部模式，并将不同的路径作为单独的微列存储在底层存储块中。这使得执行引擎在查询期间只需扫描必要的子列，从而大幅减少I/O。

> 嵌套JSON文档到物理列式存储的逻辑转换。引擎自动将路径（如`user.name`）提取到单独的存储向量 (vector)中，以优化数据获取。

这种架构意味着查询JSON对象深层特定字段的成本与查询标准原始列的成本相当，前提是路径在整个数据集中保持一致。但是，模式的高度可变性可能导致稀疏列存储，这会降低压缩率和扫描性能。

### 读取时模式（Schema-on-Read）模式

在数据仓库中保留半结构化数据的主要优点是它支持读取时模式。在传统的写入时模式（Schema-on-Write）模型中，上游数据结构的任何更改都需要目标表的迁移和ETL管道的更新。使用读取时模式，源JSON中出现的新属性会自动捕获到变体列中，而不会中断管道。

您在查询定义而非表定义中定义模式。例如，访问嵌套字段使用点表示法或括号表示法，具体取决于SQL方言：

```sql
-- 从名为 'raw_payload' 的 VARIANT 列中提取数据
SELECT 
    raw_payload:transaction_id::INTEGER as id,
    raw_payload:items[0].sku::STRING as first_item_sku,
    raw_payload:metadata.device_info.os::STRING as device_os
FROM sales_events
WHERE raw_payload:event_timestamp >= DATEADD('hour', -1, CURRENT_TIMESTAMP());
```

尽管方便，但这种灵活性带来了管理难题。如果上游应用程序将`device_info`重命名为`client_info`，上面的查询将返回`NULL`而不是明确失败。为了减少这种情况，生产系统通常采用“混合视图”层。这涉及在原始表之上创建一个SQL视图，该视图明确地将半结构化路径转换并别名为类型化列。此视图作为契约，确保下游报告工具与稳定的模式交互。

### 展平数组和层级结构

处理数组时，数据建模会变得复杂。源系统中的单个记录（例如订单）可能包含多个子实体（例如订单项）的数组。在3NF模型中，这些将是通过外键连接的单独表。在半结构化建模中，它们存在于同一行中。

要分析子实体，您必须“展平”或“取消嵌套”结构。此操作将关系从每个父实体一行转换为每个子实体一行。在关系代数中，这实际上是父行与其自身子数组之间的交叉连接。

如果我们将数据集表示为 $R$，行 $r$ 中的数组属性表示为 $r.array$，则展平后的结果集 $R_{平坦}$ 定义为：

$R_{平坦} = \{ (r_{属性}, e) \mid r \in R, e \in r.数组 \}$

大多数MPP数据库通过`LATERAL FLATTEN`（Snowflake）、`UNNEST`（BigQuery）或`PartiQL`扩展（Redshift）来实现此功能。

```sql
SELECT 
    o.raw_data:order_id::INT as order_id,
    i.value:product_id::INT as product_id,
    i.value:quantity::INT as quantity
FROM orders o,
LATERAL FLATTEN(input => o.raw_data:items) i;
```

此查询为`items`数组中的每个项目生成一行，并从父级继承`order_id`。此方法对于在子级别聚合指标非常重要，例如计算所有订单中特定产品的总销售量。

### 性能与物化

与结构化数据相比，查询半结构化数据会产生计算开销。引擎必须在运行时遍历内部结构并进行类型转换。

$总成本 = 扫描成本 + 解析成本 + 转换成本$

尽管$扫描成本$通过前面讨论的列式存储得到了最小化，但$解析成本$和$转换成本$在PB级别时可能会变得很高。如果特定路径（例如`raw_data:user_id`）用于连接条件或聚簇键，那么依赖变体路径的效率会很低。

建议的模式是物化常用路径。您可以在摄取过程中将这些字段提取到同一表内的单独的强类型列中。这会创建一个宽表结构，其中核心业务键是强类型列，而其余的负载则驻留在包罗万象的变体列中。



![查询延迟：原生列 vs. 嵌套路径](plots/8028-0.json)



> 比较性延迟分析，显示在连接和筛选等密集操作中使用嵌套路径的性能开销。当用作连接键时，原生列的性能明显优于JSON路径。

如图表所示，虽然简单选择的开销可以忽略不计，但使用JSON路径作为连接键可能会使性能降低一个数量级。发生这种情况是因为数据库无法在未类型化的数据上有效使用布隆过滤器或合并连接等标准优化，除非先对其进行解码。

### 处理演变与漂移

模式漂移发生在传入数据结构意外变化时。在半结构化场景中，这通常表现为：

1. **新字段：** 由变体列自动处理。
2. **类型更改：** 某个字段突然从发送整数变为发送字符串。
3. **结构更改：** 对象变为数组。

为管理类型更改，您的查询应使用安全的类型转换函数（例如`TRY_TO_NUMBER`或`SAFE_CAST`）。这可以防止单个格式错误记录导致整个批处理查询失败。对于结构更改，高级建模需要元数据监控。您可以对变体列运行定期分析查询，使用推断模式的函数（如Snowflake中的`INFER_SCHEMA`）来检测数据的主导结构是否发生变化，从而表明需要更新下游视图或物化。

通过结合变体类型的灵活性和物化带来的性能准确性，您可以构建一个数据模型，该模型能够抵御上游波动，同时保持高性能分析所需的速度。

## 参考资料

- [Working with nested and repeated data](https://cloud.google.com/bigquery/docs/nested-repeated) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  解释了BigQuery如何使用 STRUCT 和 ARRAY 类型原生支持嵌套数据，以及如何有效地查询和展平这些结构。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  全面介绍了数据模型、存储机制和模式演变，为在现代系统中处理半结构化数据提供了基础背景。
