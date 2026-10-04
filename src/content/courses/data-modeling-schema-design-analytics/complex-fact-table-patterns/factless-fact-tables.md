---
course: "data-modeling-schema-design-analytics"
chapter: "complex-fact-table-patterns"
lesson: "factless-fact-tables"
sourceId: 7988
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-4-complex-fact-table-patterns/factless-fact-tables"
title: "无事实的事实表"
description: "使用无事实表跟踪事件和覆盖范围。"
order: 4
plots: []
sourceHash: "427413fd83080f597d6cf0f2c949de23ff668fb44c62b1ef7c732cab005194c8"
sourceCorrections: []
---

数据架构模式通常将事实表定义为数值度量的容器。我们会寻找表示收入、数量、时长或余额的列。但是，您会遇到一些业务流程，其中主要事件不产生可衡量的指标。其价值仅在于事件本身或维度之间的关系。

这些情况需要一个**无事实的事实表**。尽管名称看似矛盾，但这种结构是维度建模中一个标准且不可或缺的部分。它捕获维度的交集，以记录事件的发生或定义覆盖范围。

### 事件跟踪表

无事实的事实表最常见的变体是记录离散事件的发生。在这些情况下，“事实”仅仅是行的存在。没有要测量的量级，只有维度在特定时间点的汇合。

设想一个学生考勤系统。如果一个学生上课，就会产生一个事件。没有“考勤量”；学生只是到场了。该表的粒度是每个学生、每门课、每天一行。

这个事件的模式看起来像一个标准的星型模式，但事实表只包含外键。

> 事件跟踪无事实表的设计。中心表完全由连接到周围维度的外键组成。

为了分析这些数据，您需要计数行。如果需要计算特定课程的总考勤数，则对按课程维度分组的行进行计数。

$\text{总考勤} = \sum_{\text{行}} 1$

在一些实现中，数据工程师会添加一个哑列（或称占位符列），通常命名为 `measure` 或 `occurrence`，其中包含常数值 $1$。这使得商业智能 (BI) 工具可以执行标准的 `SUM` 操作而不是 `COUNT`，从而简化了那些默认期望可加性指标的工具中的报告生成。

### 覆盖表

无事实的事实表的第二个应用，通常也更关键，是建模覆盖范围或条件。标准的事务事实表是稀疏的；它们只在活动发生时包含行。如果某个产品在特定日期没有销售，则 `FactSales` 表中不会有对应的行。

这种稀疏性使得回答负面问题变得困难，例如：

- “今天哪些促销产品没有销售？”
- “哪些商店授权库存零变动？”

您无法查询销售表中不存在的商品。要回答这些问题，您需要一个表来定义所有可能性，即 *应该* 发生什么或 *有资格* 发生什么。

覆盖型无事实表位于维度之间，用于定义关系。例如，一个 `FactStorePromotion` 表可能会关联 `DimProduct`、`DimStore`、`DimPromotion` 和 `DimDate`。该表中的一行表示特定产品在特定日期于特定商店进行促销，无论其是否售出。

通过将此覆盖表与销售表连接，您可以找出空白。

1. 从**覆盖表**中选择所有预期行（产品 A、商店 B、日期 1）。
2. 使用对应的键与**销售事实表**进行左连接。
3. 筛选销售指标为 `NULL` 的行。

这种技术对于库存分析、保单覆盖（即使没有索赔也对有效保单进行建模）以及销售区域分配很有用。

### 主要方法

在标准事实表中，主键通常是所有外键的复合。在无事实的事实表中，这条规则通常也适用，但您必须确保粒度是唯一的。

对于考勤示例，`Student_SK`、`Course_SK` 和 `Date_SK` 的组合保证了唯一性，因为一个学生不能在一天内上同一节课两次。但是，如果您正在跟踪系统登录（一个事件无事实表），一个用户可能一天登录多次。在这种情况下，用户和日期的复合键不是唯一的。您要么需要在粒度中添加一个时间维度，要么为事实表本身引入一个代理键，以唯一地标识每个事件。

### 何时使用无事实表

当业务流程不产生自然度量，但维度的交集能提供分析价值时，您应该引入这种模式。

- **跟踪工作流转换：** 记录文件从“草稿”状态变为“审查”状态。
- **分配：** 将销售人员与区域关联，或将客户与细分群体在特定时期内关联。
- **资格：** 记录哪些员工有资格获得特定的福利计划。

认识到事实不严格要求数字，这使您能够对业务环境的全貌进行建模，不仅记录发生了什么（事务），还记录这些事件发生的背景和条件。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling, 3rd Edition](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/data-warehouse-business-intelligence-toolkit-books/data-warehouse-toolkit-3rd-edition/) — Ralph Kimball, Margy Ross, Joy Mundy, Bob Becker, and Warren Thornthwaite (2013)
  Publisher: John Wiley & Sons
  这本关于维度建模的权威指南全面介绍了无事实事实表，包括事件跟踪和覆盖范围分析。
- [Factless Fact Tables](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/factless-fact-tables/) — Ralph Kimball (2004)
  Journal: TDWI (The Data Warehousing Institute); Publisher: The Kimball Group
  这是一篇来自维度建模创始人之一的简明文章，直接解释了无事实事实表的概念及其应用。
