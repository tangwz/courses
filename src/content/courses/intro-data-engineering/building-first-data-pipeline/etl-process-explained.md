---
course: "intro-data-engineering"
chapter: "building-first-data-pipeline"
lesson: "etl-process-explained"
sourceId: 5526
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-3-building-first-data-pipeline/etl-process-explained"
title: "ETL 过程说明"
description: "了解提取、转换、加载（ETL）数据集成模式中包含的步骤。"
order: 2
plots: []
sourceHash: "1f7b787187d45626d9ee59a04e82497d43aea6eb11b1a7f215a0fdced2c227ea"
sourceCorrections: []
---

ETL 指的是提取、转换、加载。它是一种成熟的模式，用于数据管道中，以从不同来源收集数据，重塑数据，然后将其存入指定的目标系统，通常是数据仓库。可以将其视为在烹饪之前准备食材（提取和转换），然后再将其放入最终菜肴（加载）。ETL 的主要特点是数据转换发生在数据到达其最终目的地*之前*。

我们来逐一分析每个步骤：

### 提取 (E)

第一步是将数据从其原始源中取出。数据可能存在于许多地方：

- **数据库：** 关系型数据库（如 PostgreSQL、MySQL）或 NoSQL 数据库。提取通常涉及编写查询（例如，使用 SQL）来选择所需的数据。
- **API（应用程序编程接口）：** 许多网络服务提供 API 以编程方式访问其数据。提取涉及向这些 API 发送请求。
- **文件：** 数据可能存储在平面文件中，如 CSV（逗号分隔值）、JSON（JavaScript 对象表示）或应用程序生成的日志文件。提取涉及读取这些文件。
- **流式来源：** 实时数据流，尽管通常由不同的模式处理，但有时也可以批量处理用于 ETL。

提取阶段的目标是从一个或多个这些源中获取必要的原始数据。此处效率很重要。您只想获取下游步骤所需的数据，以避免不必要的处理负担。

### 转换 (T)

这通常是 ETL 过程中最复杂的部分。数据提取后，通常需要修改才能变得有用且一致。转换步骤获取原始数据并应用一组规则或函数来为目标系统准备数据。这通常发生在临时暂存区或内存中。常见的转换操作包括：

- **清洗：** 修正错误、处理缺失值（例如，用默认值或平均值替换）、删除重复项以及标准化格式（例如，确保所有日期都采用 `YYYY-MM-DD` 格式）。
- **结构化：** 将复杂数据格式（如 JSON 或 XML）解析成更结构化的格式，如行和列。
- **丰富：** 合并来自不同来源的数据。例如，将 CRM 系统中的客户信息与电子商务平台中的销售数据合并。
- **聚合：** 汇总数据。例如，计算每个产品类别的总销售额或统计每天的网站访问量。
- **筛选：** 只选择符合特定条件的行或列。
- **拆分/合并：** 将单列拆分为多列或将多列合并为一列。

具体的转换完全取决于目标系统的要求和数据的预期用途（例如，用于报告或分析）。

### 加载 (L)

数据转换完成后，最后一步是将其加载到目标系统。该目标通常是为分析而优化的结构化存储库，例如：

- **数据仓库：** 一个为商业智能和报告设计的中央存储库。
- **数据集市：** 数据仓库的一个更小、更集中的子集，通常专用于特定部门或主题。
- **数据库：** 有时数据会加载回操作数据库以供应用程序使用，尽管对于 ETL 而言，加载到分析系统更为常见。

加载数据有不同的策略：

- **全量加载：** 这涉及删除目标表中所有现有数据并从头开始重新加载所有数据。它很简单，但对于大型数据集来说可能速度慢且资源密集。
- **增量加载（或差异加载）：** 这涉及仅加载自上次加载以来已更改或新增的数据。这对于大型数据集更高效，但需要跟踪更改的机制（例如，使用时间戳或变更数据捕获）。

以下图表描绘了 ETL 过程的典型流程：

> 一张图表展示了数据如何从各种源系统，通过 ETL 过程中的提取、转换和加载阶段，最终到达目标系统。

ETL 管道在您需要执行复杂数据清洗、集成来自多个异构源的数据，并将其加载到高度结构化的环境（如数据模式预先定义明确的关系型数据仓库）时特别有用。重点在于在数据到达其最终存储位置之前进行彻底准备。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball and Margy Ross (2013)
  Publisher: Wiley
  这本基础著作阐述了数据仓库和ETL设计的原理，详细涵盖了抽取、转换和加载过程的实践方面。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  讨论数据系统的核心概念，包括批处理、数据集成以及与ETL相关的目标分析存储的特点。
- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108182/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  提供数据工程的现代视角，解释ETL模式如何融入现代数据管道和架构方法。
