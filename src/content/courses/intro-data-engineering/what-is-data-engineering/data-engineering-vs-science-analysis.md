---
course: "intro-data-engineering"
chapter: "what-is-data-engineering"
lesson: "data-engineering-vs-science-analysis"
sourceId: 5490
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-1-what-is-data-engineering/data-engineering-vs-science-analysis"
title: "数据工程 vs 数据科学 vs 数据分析"
description: "了解数据工程、数据科学和数据分析角色之间的差异与合作。"
order: 3
plots: []
sourceHash: "0e8e0661a8118e82e12ef4f500f9c39b9a683d55f8acc0a09b979190a38a055f"
sourceCorrections: []
---

在不断扩大的数据行业中，您会经常听到数据工程、数据科学和数据分析这些术语。虽然它们紧密合作，通常为了相似的组织目标，但它们的工作重点、方法和日常任务有显著差异。明白这些差异对于理解数据工程的特定作用非常重要。可以把它们想象成在一个大型项目上协作的不同专业人员，每个人都贡献自己的独特技能。

### 数据工程：构建者

正如本章开始介绍的，数据工程师是数据架构的设计师和建造者。他们的主要职责是设计、构建和维护大规模数据处理所需的基础设施和数据管道。

- **目标：** 保证数据系统的可靠性、效率和可访问性。他们确保数据能从源头到达需要的地方，格式正确，并得到妥善存储。
- **工作重点：** 基础设施、数据管道（ETL/ELT 流程）、数据库、数据仓库、数据湖、可伸缩性、可靠性、系统级别的数据质量检查。
- **常见问题：** 我们如何有效地从这些不同来源收集数据？我们应该如何存储这个庞大的数据集？我们如何将这些原始数据转换为可用格式？数据管道运行可靠吗？这个系统明年能处理两倍的数据量吗？
- **产出：** 可用的数据管道、结构良好的数据存储系统（如数据仓库或数据湖）、干净且可访问的可用数据集。

### 数据分析：解读者

数据分析师使用工程师准备的数据（有时也处理结构化程度较低的数据），并专注于从中获取有价值的见解。他们审视历史数据以找出趋势、回答特定的业务问题，并通过报告和可视化来传达发现。

- **目标：** 理解过往表现，找出趋势，并向业务利益相关者提供可操作的见解。
- **工作重点：** 查询数据（通常使用 SQL）、数据清洗（在分析层面）、描述性统计、数据可视化、报告、仪表板创建、沟通。
- **常见问题：** 我们上季度的销售额是多少？哪个营销活动表现最好？我们客户群的主要特点是什么？你能创建一个显示网站流量的仪表板吗？
- **产出：** 报告、仪表板、可视化、发现摘要、基于现有数据对特定业务查询的回答。

### 数据科学：预测者和实验者

数据科学家通常使用准备好的数据来展望未来或找出更复杂的数据模式。他们应用统计技术、机器学习 (machine learning)算法和实验设计来构建预测模型、分类数据或理解复杂行为。

- **目标：** 进行预测、构建分类模型、设计实验，并找出通过简单分析可能不明显的复杂模式。
- **工作重点：** 统计学、机器学习算法、预测建模、实验设计、编程（通常是 Python 或 R）、特征工程、算法优化。
- **常见问题：** 我们能否预测哪些客户可能流失？我们如何自动分类传入的支持工单？哪些因素影响用户参与度？这项新功能是否提高了转化率（A/B 测试）？
- **产出：** 预测模型、分类工具、实验结果、更详尽的统计见解、可集成到应用程序中的算法。

### 它们如何配合工作

这些角色高度相互依赖。数据工程师提供根本支撑。如果没有可靠的数据管道和存储，分析师将难以获取所需数据，科学家也无法获得复杂建模所需的数据质量或数量。

> 这张图展示了数据工程、数据分析和数据科学角色之间典型的数据流和交互。数据工程师负责收集和准备数据，使得分析师和科学家能够从中获取价值。

分析师可能会发现工程师需要在数据管道上游修复的数据质量问题。科学家可能需要新的数据源或特定的数据特征，这会促使工程师修改现有管道或构建新管道。分析师的见解和科学家的模型经常为数据基础设施产生新的要求。

在小型组织中，一个人可能身兼多职，执行工程、分析和科学方面的任务。然而，随着数据量和复杂性的增加，专业化变得不可或缺。理解每个角色的核心关注点有助于阐明为什么数据工程在现代数据体系中是如此独特且举足轻重的一个专业。它为数据分析和数据科学的工作提供了必要的支持。

## 参考资料

- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  本书清晰介绍了数据工程领域，系统地定义了其原则，并区分了其与数据分析和数据科学的角色。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38870-1](https://doi.org/10.1007/978-3-031-38870-1)
  一本备受推崇的教科书，涵盖了统计学习和机器学习的基本概念，对于理解数据科学中采用的方法至关重要。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  深入回顾了构建和维护数据系统所需的工程原则，为数据工程师创建的基础设施和数据管道提供了背景。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling (3rd Edition)](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  理解数据仓库概念和维度建模的基石资源，这对数据分析师和科学家使用的数据结构至关重要。
