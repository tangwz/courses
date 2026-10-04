---
course: "data-modeling-schema-design-analytics"
chapter: "physical-design-optimization"
lesson: "one-big-table-obt-methodology"
sourceId: 7993
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-5-physical-design-optimization/one-big-table-obt-methodology"
title: "单一巨表 (OBT) 方法"
description: "OBT设计模式的优劣分析"
order: 4
plots: ["plots/7993-0.json"]
sourceHash: "134d602430ba656926ffe3f279f8e2667d813937cd3ef364c00a19e5f63ac67c"
sourceCorrections: []
---

单一巨表方法（简称OBT）代表了反范式化的一个极端。某些数据建模方式，比如星型和雪花型模式，试图兼顾范式化和查询性能，而OBT方式则完全不区分事实和维度。它将所有所需的数据属性归拢到一张极宽的表中，其中既有度量指标也有描述性属性。

随着Google BigQuery、Snowflake和Amazon Redshift等列式存储引擎的兴起，这种设计模式受到广泛应用。传统行式数据库难以应对此类表的冗余和宽度问题，而列式存储则具有独特的架构，能高效地处理它们。

### OBT的架构

经典的星型模式下，查询通常需要连接中心事实表与多个维度表，才能得到易读的属性。比如，要按产品类别和区域报告销售情况，数据库引擎必须扫描销售事实表，将其与产品维度连接，再与区域维度连接。

OBT架构在数据加载（ETL/ELT）过程中就预先完成了这些关联的连接。最终形成的表包含事务数据以及所有对应维度的文本描述。

> 结构上的对比，展示了星型模式中的关联关系如何在OBT方法中被整合为单一结构。

### 性能机制：存储与计算的权衡

采用OBT的主要目的是将成本从计算（CPU）转移到存储（磁盘/IO）。在分布式计算环境中，连接操作开销很大。它们常涉及**数据混洗**，即数据需在不同节点间通过网络传输，以匹配键值后才能进行连接。

使用OBT可以消除查询时的连接阶段。查询引擎只需扫描表并汇总结果。

#### 压缩与列式存储

OBT的批评者常提及数据冗余问题。如果一个客户名称在100万条事务记录中重复出现，存储该字符串100万次似乎是低效的。但现代列式数据仓库采用高级编码技术，特别是行程编码（RLE）和字典编码。

如果`Customer_Name`列中的值“Acme Corp”连续重复1,000次，存储引擎不会存储该字符串1,000次。它只存储该值一次，并记录它在接下来的1,000行中重复。因此，列式数据库中反范式化带来的存储开销远低于传统关系型系统。

我们可以用查询复杂度来表示性能提升。若$C_{连接}$是表连接的开销，$C_{扫描}$是读取数据的开销：

$Cost_{星型} \approx C_{扫描(事实)} + \sum C_{扫描(维度)} + C_{连接}$

$Cost_{OBT} \approx C_{扫描(宽表)}$

由于$C_{连接}$通常随数据量非线性增长（因排序和混洗），移除它能让大型数据集上的查询性能保持稳定。

### 权衡考量

OBT能为许多分析查询带来更好的读取性能，但它也带来数据管理和灵活度方面的特殊问题。

#### 1. 重建延迟

当某个维度发生变化时，比如产品名称被修改，星型模式只须更新维度表中的一行数据。而在OBT设计中，该产品名称可能存在于巨表中的数百万行里。要体现此更改，必须重新加载或更新该产品的所有出现位置。这使得OBT不太适合处理经常变化的数据（高变动维度），除非每天进行全表重新加载。

#### 2. 模式演进

在星型模式中，给维度增加一个新属性属于轻量级的元数据操作。而在OBT中，增加一列则需要修改一张庞大的表。尽管许多现代数据仓库能高效处理这种元数据变更，但用历史数据回填新列可能非常耗费资源。

#### 3. 宽度限制

尽管云数据仓库支持非常宽的表，但列数存在硬性限制（通常在1,000到10,000之间）。如果你聚合的维度数量超过这些限制，OBT在技术上将无法实现。

### 查询执行的可视化

为明了效率的提升，请看查询执行期间的时间分配。针对星型模式的查询，其大部分时间耗费在“混洗”和“连接”阶段。OBT则尽量缩短这些阶段，将主要时间用于扫描和汇总。



![查询执行时间分配](plots/7993-0.json)



> 查询执行期间资源分配的对比。OBT去除了混洗/连接的额外开销，但由于读取更宽的表结构，扫描时间可能略有增加。

### 实施策略

实际工作中，数据工程师很少将OBT作为主要事实来源。更好的做法是将其作为范式化层的一个下游产物来实施。你可以维护一个整洁的星型模式或3NF模型以确保数据完整性和治理，然后专门为Tableau、PowerBI或Looker等消费层生成OBT。

这种模式涉及创建一个视图或物化表，以连接事实表和维度表：

```sql
CREATE OR REPLACE TABLE analytics.sales_flat_model AS
SELECT
    f.transaction_id,
    f.transaction_date,
    f.amount,
    p.product_name,
    p.category,
    c.customer_segment,
    r.region_name
FROM fact_sales f
LEFT JOIN dim_product p ON f.product_id = p.product_id
LEFT JOIN dim_customer c ON f.customer_id = c.customer_id
LEFT JOIN dim_region r ON f.region_id = r.region_id;
```

将OBT视为“服务层”而非“存储层”，既能为最终用户带来性能上的优势，又能保持上游管道中关系模型的易于管理。这种混合方式让分析团队可以兼顾维护效率和查询速度。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling, 3rd Edition](https://books.google.com/books/about/The_Data_Warehouse_Toolkit.html?id=P76XAQAAQBAJ) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons; Pages: 608
  描述了维度建模的原则，为OBT去范式化所对比的范式化提供了背景。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了数据存储、处理和系统设计的基本原则，有助于理解各种数据模型（包括高度去范式化结构）的架构优势和挑战。
- [Optimizing schema design for BigQuery](https://cloud.google.com/bigquery/docs/best-practices-performance-overview) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  提供了关于利用BigQuery列式架构进行分析工作负载的官方指导，包括去范式化策略。
- [Designing tables for performance](https://docs.snowflake.com/en/user-guide/tables-design-performance) — Snowflake Documentation (2024)
  提供了优化Snowflake架构表结构的指导，包括考虑去范式化以提高查询速度。
