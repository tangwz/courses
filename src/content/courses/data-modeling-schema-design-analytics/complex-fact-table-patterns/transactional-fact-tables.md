---
course: "data-modeling-schema-design-analytics"
chapter: "complex-fact-table-patterns"
lesson: "transactional-fact-tables"
sourceId: 7985
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-4-complex-fact-table-patterns/transactional-fact-tables"
title: "事务事实表"
description: "在事务事实表中建模离散事件。"
order: 1
plots: ["plots/7985-0.json"]
sourceHash: "4d96a8ee5550a0a0e280b537850c852728fffd4306138585e8633c2bca76d716"
sourceCorrections: []
---

事务事实表在维度建模中代表最基础的粒度。它们对应于时间点发生的事件。当客户进行购买、机器记录错误或用户点击链接时，操作系统会记录事务。这些独立的动作中的每一个都会在事实表中生成一行。

事务事实表的主要特点是其粒度。粒度是可用的最细致的细节级别。对于零售业务，粒度通常是收据上每一行商品对应一行，而不仅仅是每张收据一行。如果客户在一个购物篮中购买了五种不同的产品，事务事实表会记录五条独立的行。这种细致的级别使得分析最为灵活，因为你可以将数据汇总到业务所需的任何聚合级别。

### 事务事实表的结构

事务事实表由两种主要类型的列组成：外键和度量。外键将事件关联到相关的维度表，提供谁、什么、何地以及何时等背景信息。度量是事件生成的数值数据点，例如销售数量或总金额。

有时，你会遇到一些数据元素，它们作用类似维度，但由于其事务独有性，不适合放入独立的维度表。发票编号或事务ID就是典型例子。这些被称为退化维度。它们保留在事实表中，以便按父事务进行分组，同时避免连接到庞大、低价值维度表的额外开销。

下图展示了一个标准销售事务模式。请注意中心事实表如何存储度量值并关联到描述性维度。

> 一个事务事实表连接到四个维度的结构视图。Fact\_Sales表包含外键、退化维度和可加性度量。

### 可加性与稀疏性

事务事实表天然具有可加性。这意味着其中包含的度量可以在所有关联维度上进行求和。这一特性大大简化了分析。你可以按天、按商店、按产品类别或按客户群对`Sales_Amount`进行求和，而无需应用复杂的业务逻辑。

例如，要计算总收入，算术操作非常简单：

$总收入 = \sum (\text{销售金额})$

这与快照表（如银行余额）中的半可加性度量不同，后者无法在时间维度上求和。

另一个显著特点是稀疏性。事务事实表是稀疏的，因为它们只包含实际发生的事件对应的行。如果某个产品在特定日期没有销售，那么表中不会有该产品-日期组合的行。不会插入“零”行来表示没有活动。这种效率使得表能够扩展到数十亿行，同时保持聚合查询的高性能。

### 分析数据

当查询事务事实表时，你几乎总是在执行聚合操作。虽然原始表存储的是单个商品项级别的数据，但业务报告通常需要汇总视图。

考虑一个分析师需要了解特定月份按产品类别划分的销售业绩的场景。该查询将细粒度的事务行聚合为更高级别的度量。

```sql
SELECT 
    d.Category,
    SUM(f.Quantity) as 总单位数,
    SUM(f.Sales_Amount) as 总收入,
    COUNT(DISTINCT f.Invoice_Number) as 总订单数
FROM Fact_Sales f
JOIN Dim_Product p ON f.Product_Key = p.Product_Key
JOIN Dim_Date d ON f.Date_Key = d.Date_Key
WHERE d.Month = '2023-10'
GROUP BY p.Category;
```

在此查询中，`COUNT(DISTINCT f.Invoice_Number)`利用退化维度来统计唯一的购物篮，而`SUM`操作提供总数量和总价值。从这个聚合视图深入到单个事务的能力是这种模式的一个主要优点。

### 处理增长和分区

由于事务事实表记录每个独立的事件，它们会快速增长。在高数据量环境（如网站分析或物联网传感器日志）中，这些表每小时可以摄取数百万行。尽管现代云数据仓库可以处理海量数据集，但高效的物理设计对于保持查询速度是必要的。

事务表最常见的优化措施是按时间分区。由于大多数分析查询都按日期范围（例如“过去30天”或“今年至今”）筛选，在日期列上对表进行物理分区可以使数据库引擎只扫描相关文件。



![查询性能：扫描数据与分区](plots/7985-0.json)



> 分区表与非分区表在查询期间扫描数据量的比较。分区确保扫描的数据量与请求的时间范围成比例。

如图所示，如果不分区，查询一天的数据可能扫描整个表的历史数据。而通过分区，扫描仅限于该特定日期的数据。

### 何时选择此模式

在以下情况使用事务事实表：

1. 你需要以最细致的级别分析单个事件。
2. 度量值天然具有可加性（求和、计数）。
3. 你需要能够按关联维度的任意组合对数据进行分组。
4. 事件的缺失意味着没有活动（稀疏性）。

虽然事务事实表非常灵活，但并非适用于所有问题。如果你需要分析实体的状态（如当前库存水平）或完成某个过程所需的时间（如订单履行时长），你会发现从原始事务计算这些状态需要复杂且开销大的子查询。对于这些情况，我们转向快照事实表和累积快照事实表，我们将在后续章节中介绍它们。

## 参考资料

- [Star Schema The Complete Reference](https://books.google.com/books/about/Star_Schema_The_Complete_Reference.html?id=oG0NtwAACAAJ) — Christopher Adamson (2010)
  Publisher: McGraw-Hill Education; Pages: 486
  星型模式设计的详细解释，包括事实表和维度表的结构及关系。
- [Data Warehouse Design: Modern Principles and Best Practices](https://www.mheducation.com/highered/product/data-warehouse-design-rizzi-varesio-golfarelli/9781260455855.html) — Stefano Rizzi, Luca Varesio, Matteo Golfarelli (2018)
  Publisher: McGraw-Hill Education
  探讨现代数据仓库设计原则，包括大型事实表的分区等物理设计考量。
