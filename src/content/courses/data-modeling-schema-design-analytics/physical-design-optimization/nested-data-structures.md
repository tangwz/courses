---
course: "data-modeling-schema-design-analytics"
chapter: "physical-design-optimization"
lesson: "nested-data-structures"
sourceId: 7992
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-5-physical-design-optimization/nested-data-structures"
title: "嵌套数据结构"
description: "在现代数据仓库中使用数组和结构体进行建模。"
order: 3
plots: ["plots/7992-0.json"]
sourceHash: "d91799aa4753dffcce8d4be31097821eceb5517ffebc082ee2692fdcfedf1e4f"
sourceCorrections: []
---

几十年来，第一范式 (1NF) 在数据库设计中一直是一项不可动摇的规则。它要求每个列都必须包含原子值，这意味着单个单元格不能存储列表或属性集。为了表示一对多关系，例如包含多个行项目的电子商务订单，您需要创建两个独立的表并通过外键将它们关联起来。

现代云数据仓库已经从根本上改变了这一要求。通过支持半结构化数据类型，特别是数组（Arrays）和结构体（Structs），像 Google BigQuery、Snowflake 和 Databricks 这样的系统允许您将一对多关系直接反范式化到一个表中，而无需复制父行。这种方法通常被称为带有嵌套结构的“写入时模式”，它提供了一种有效方式来通过避免开销大的连接来提升查询性能。

### 嵌套数据的运作方式

要了解何时使用嵌套结构，我们首先需要明确大多数分析引擎中可用的两种主要复杂数据类型。

**数组（重复字段）**
数组是相同数据类型的有序值列表。在标准关系模型中，如果您想为一个商品存储一系列产品标签（例如，“电子产品”、“促销”、“新品”），您会创建一个 `ProductTags` 桥接表。使用数组，您只需定义一个 `ARRAY<STRING>` 类型的列。数据库会将这些值与父行连续存储。

**结构体（记录/对象）**
结构体是一个容器，它将相关字段以单一名称分组在一起。它的功能类似于 JSON 对象或字典。例如，您不必为 `shipping_street`、`shipping_city` 和 `shipping_zip` 设置单独的列，而是可以创建一个包含这些子字段的 `shipping_address` 结构体。

当这些数据类型结合使用时，它们真正的优势就体现出来了。一个**结构体数组**允许您将子表的整个行嵌套到父表中。

考虑一个电子商务数据库的结构。在范式化设计中，“订单”表与“订单项”表关联。在嵌套设计中，“订单项”直接嵌入 (embedding)在“订单”行中。

> 范式化的一对多关系与子记录位于父行内的嵌套结构的比较。

### 性能影响

在分析场景中使用嵌套数据的主要原因是减少 I/O。在分布式数据库中，关联两个庞大的表（如 `Orders` 和 `LineItems`）通常会触发“数据混洗”。引擎必须跨网络移动数据，以便在同一计算节点上对齐 (alignment)匹配的键。此操作网络开销大且速度慢。

通过将订单项嵌套在订单中，您可以强制实现**数据局部性**。子数据在磁盘上物理存储在父数据旁边。当您查询订单时，订单项会预先加载，无需进行关联或网络混洗。

然而，这并不意味着数据库存储的是原始 JSON 字符串（这会解析缓慢）。像 Parquet 和 Capacitor（BigQuery 使用）这样的现代格式会将这些嵌套结构分解成独立的子列。

如果您有一个包含嵌套字段 `items.price` 的 `Orders` 表，存储引擎会为 `items.price` 创建一个特定列。如果您的查询计算总收入，引擎将 *只* 扫描 `items.price` 子列。它会完全跳过 `items.product_name` 或 `items.description` 子列。

我们可以从数据访问的角度来表示避免关联所带来的效率提升。在范式化关联中，成本函数 $C$ 大致与两个关系的规模加上混洗开销相关：

$C_{关联} \approx \text{扫描}(R) + \text{扫描}(S) + \text{混洗}(R, S)$

使用嵌套结构时，成本会降低到仅扫描所需的特定子列，且没有混洗开销：

$C_{嵌套} \approx \text{扫描}(R_{父列}) + \text{扫描}(R_{嵌套子列})$

### 查询嵌套数据

尽管嵌套改进了存储和读取性能，但它增加了编写 SQL 的复杂程度。您不能简单地选择包含数组的列；您必须“展开”或“扁平化”它才能与单个元素交互。

当您 `UNNEST` 一个数组时，查询引擎会在查询期间在父行和数组元素之间临时创建一个笛卡尔积。这模仿了关联的结果，但完全在单个节点的内存中发生，从而保持了高效率。

例如，要筛选包含特定商品的订单，您可能需要编写查看数组结构内部的逻辑。这需要熟悉您的数据仓库特有的语法（例如 BigQuery 中的 `CROSS JOIN UNNEST` 或 Snowflake 中的 `LATERAL FLATTEN`）。

### 嵌套的策略性使用

嵌套并非星型模式的通用替代方案。它是一种特定的优化工具。您应根据自己的访问模式评估其优缺点。

#### 何时使用嵌套结构

1. **父子局部性：** 子数据几乎总是在父数据的上下文 (context)中被访问。例如，您很少在不了解订单项所属订单的情况下分析 `LineItems`。
2. **高基数关联：** 父表和子表之间的关联规模庞大，导致性能瓶颈。
3. **不可变事件日志：** 数据表示固定的事件，例如具有重复参数 (parameter)的 Web 日志或应用程序错误跟踪。

#### 何时避免嵌套

1. **高波动性：** 更新数组中的单个元素（例如，更改一个订单项的数量）需要重写整个父分区或文件。这对于频繁更新来说效率低下。
2. **多对多关系：** 如果一个子实体（如“产品”）被多个父实体共享，并且有其自身的会变化的属性（如“产品名称”），那么嵌套它会导致数据重复。如果产品名称发生变化，您将不得不更新所有包含该产品的订单。
3. **BI 工具兼容性：** 一些传统商业智能工具能很好地处理扁平表，但在不先创建自定义 SQL 视图将其扁平化的情况下，难以解析或可视化嵌套数组。

下图突出了基于数据关系复杂性和更新频率的决策边界。



![嵌套结构的适用性](plots/7992-0.json)



> 更新频率低且查询耦合度高的场景从嵌套中获益最大，而频繁更新、耦合度松散的数据在范式化表中能保持更好的性能。

通过合理运用嵌套数据结构，您可以降低关联逻辑的复杂程度，并显著减少分析查询的延迟。这种技术在文档存储的灵活性和列式仓库的分析能力之间搭建了一座桥梁。

## 参考资料

- [Nested and repeated columns (ARRAYs and STRUCTs)](https://cloud.google.com/bigquery/docs/nested-repeated) — Google Cloud (2024)
  Publisher: Google Cloud
  BigQuery 中使用数组和结构体的官方指南，包括查询和性能优化实践。
- [Semi-structured Data: Introduction](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpK7lFg6b09xGRMRPoZjEvrRFWk_U5xwImlyi6lXgV28dH0MQui6Wf5h2DKMxmcVowy9BLkDCXA0PT224VF9znXbuY42dPMO6OvY7H_sKvuxe-NknttoJI-DBDHfWdRAuW7oCy-Jtd-IVpiTN6OgU5o10QPIuI) — Snowflake Inc. (2018)
  Publisher: Snowflake Inc.
  Snowflake 官方文档，介绍半结构化数据类型（如 ARRAY 和 OBJECT）以及 LATERAL FLATTEN 函数。
- [Database System Concepts](https://www.db-book.com/db7/index.html) — Abraham Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education; Pages: Chapter 7: Relational Database Design
  一本广受尊敬的教科书，提供关于数据库范式化（包括第一范式 1NF）的全面学术基础，对于理解传统背景至关重要。
- [Apache Parquet](https://parquet.apache.org/) — Apache Software Foundation (2024)
  Apache Parquet 的官方文档，这是一种广泛使用的列式存储格式，支持复杂的嵌套数据结构，是许多现代数据仓库的基础。
