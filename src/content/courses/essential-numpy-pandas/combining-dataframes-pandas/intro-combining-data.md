---
course: "essential-numpy-pandas"
chapter: "combining-dataframes-pandas"
lesson: "intro-combining-data"
sourceId: 1830
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/intro-combining-data"
title: "数据合并介绍"
description: "了解何时需要组合来自不同来源或 DataFrame 的数据。"
order: 1
plots: []
sourceHash: "5f999d2ced57e5fb92b25589d4e196932e8d9de70464bcb14632f8f0b8e1a996"
sourceCorrections: []
---

正如我们在本章开篇提到的，数据很少整齐地打包在一个文件中或一张表中。更多时候，您需要的信息分散在多个来源。客户详细信息可能与他们的订单历史分开存储，或者实验结果记录在不同时间段的不同文件中。为了获得完整的数据视图并进行有意义的分析，您需要方法将这些分散的数据整合起来。

请考虑以下常见情况：

- **客户分析：** 设想一个 `DataFrame` 包含客户人口统计信息（如ID、姓名、地点），另一个包含交易记录（如客户ID、购买产品、日期、金额）。要了解哪些人口统计特征与特定购买行为相关联，您需要使用共同的 `客户ID` 将这两个 DataFrame 连接起来。
- **时间序列聚合：** 您可能会收到存储在单个文件中的每日传感器读数（例如，`readings_2023-10-26.csv`、`readings_2023-10-27.csv`）。为了进行周或月度分析，您需要将这些单独的每日 DataFrame 堆叠成一个更大的 `DataFrame`。
- **特征丰富：** 您可能有一个关于产品的主数据集，以及包含附加特征的补充数据集，例如产品类别信息或供应商详情。合并这些数据可以让您的主数据集获得更多背景信息。

数据通常不会整齐地存储在单个文件或表格中，而是分散在多个来源。为了完整地查看和有意义地分析数据，需要能够智能地组合 `DataFrame` 对象的工具。当目标是基于共享信息或结构对齐 (alignment)和集成数据时，简单地执行逐元素加法或逐个添加列通常是不够的。

本章将介绍专门为组合数据集设计的主要 Pandas 函数：

1. **拼接 (`pd.concat`)**：这对于将数据集相互堆叠（追加行）或并排放置（添加列）非常有用。可以将其想象成沿着一个轴将表格粘合在一起。
2. **合并与连接 (`pd.merge`, `.join`)**：这些方法执行数据库样式的连接。它们根据一个或多个共享列（称为键）中的值或基于 DataFrame 索引来组合数据集。当您需要链接来自不同表的相关信息时，例如将客户ID与交易关联起来，这非常必要。

理解何时以及如何使用这些技术是数据准备和分析中的一项基本技能。它使您能够从分散的来源构建统一的数据集，从而进行更全面和有见地的分析。以下章节将详细说明拼接和合并的工作原理，并通过实际示例说明其用法。

## 参考资料

- [pandas User Guide: Merge, join, concatenate and compare](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) — The pandas development team (2023)
  Pandas 官方文档，包含 `concat`、`merge` 和 `join` 等函数的详细说明和示例，对组合数据框至关重要。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由 Pandas 创始人撰写的全面书籍，提供了数据操作的广泛指导，包括数据框的组合和重塑。
- [Hands-On Data Analysis with Pandas: A Python data science handbook for data collection, wrangling, analysis, and visualization](https://www.packtpub.com/product/hands-on-data-analysis-with-pandas-second-edition/9781800563452) — Stefanie Molin (2021)
  Publisher: Packt Publishing
  使用 Pandas 进行数据分析的实用指南，涵盖了数据准备技术，如数据的组合、清理和转换，以进行分析。
