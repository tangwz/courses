---
course: "data-modeling-schema-design-analytics"
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-5-physical-design-optimization"
sourceId: 1393
chapter: "physical-design-optimization"
title: "物理设计与优化"
order: 5
description: "使用分区、聚类、嵌套数据类型和一张大表（OBT）策略来优化模式性能。"
hasQuiz: false
---

逻辑数据模型为数据分析提供抽象结构，但物理实现决定了实际的查询延迟和成本。一个设计良好的模式必须考虑数据库引擎如何从磁盘存储和检索数据块。本节讲述了在列式数据仓库中优化存储布局所需的特定配置。

我们首先考察分区策略。通过将大型表基于特定列（通常是日期）分割成更小的段，您可以使查询引擎在检索时跳过数据集的大部分。如果一个表有 $T$ 总字节，并且您查询代表一小部分数据的单日数据，分区剪枝会大幅降低扫描成本。对于一个完美剪枝的查询，我们可以将理想的扫描量 $V$ 表示为：

$$V_{scanned} \approx \frac{V_{total}}{N_{partitions}}$$

接下来，我们查看排序和聚类。尽管分区粗略地分割数据，但聚类将这些段中的行组织起来，以提高过滤谓词的效率。您将学习如何选择合适的列进行聚类，以最小化I/O操作。

本课程也涉及半结构化数据的处理。现代数据仓库允许使用嵌套字段，例如数组和结构体，这可以简化数据摄取并降低连接逻辑的复杂程度。最后，我们评估一张大表（OBT）设计。这种方法通过完全消除连接，将维度和事实展平到一个宽表中，从而以牺牲存储效率换取读取性能。您将评估这种高度非规范化方法与传统星型模式之间的权衡。
