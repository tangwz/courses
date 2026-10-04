---
course: "intro-data-engineering"
chapter: "what-is-data-engineering"
lesson: "common-data-engineering-tasks"
sourceId: 5496
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-1-what-is-data-engineering/common-data-engineering-tasks"
title: "常见数据工程任务"
description: "了解数据工程师的日常工作内容概览。"
order: 5
plots: []
sourceHash: "191c3b7113997acb6a578dcf9932d867296656ad754dd615a0bdb7189c5a0442"
sourceCorrections: []
---

数据工程师的日常工作内容包括一系列常见的任务。这些任务主要围绕构建、维护和优化数据处理系统，确保数据可靠、可用，并能被分析师、数据科学家以及AI模型等应用程序使用。可以将数据工程师视为数据的建造者和管道工；他们确保数据流动顺畅，并在需要时保持干净和可用。

以下是数据工程师最常见的一些任务：

### 设计和构建数据管道

这通常被认为是核心职责。数据工程师设计并搭建数据传输路径，即数据管道，实现数据的自动移动和转换。这包括：

- **提取**数据，从各种来源，如数据库、应用程序日志、API（应用程序编程接口）或外部供应商。
- **转换**原始数据，通过清洗（处理缺失值、纠正错误）、结构化（解析JSON或XML）、丰富化（与其他数据合并）以及将其转换为适合分析或存储的格式。
- **加载**处理过的数据到目标系统，这可以是数据库、用于报告的数据仓库，或者用于大规模存储的数据湖。
  您经常会听到ETL（提取、转换、加载）和ELT（提取、加载、转换）这两个缩写来描述这些管道模式。我们将在第3章中更详细地介绍这些内容。

### 数据摄取和收集

数据在移动或转换之前，需要先进入系统。数据工程师会设置流程，从数据的起始点收集数据。这可能涉及编写脚本定期从API拉取数据，配置工具实时传输传感器数据，或者设置连接以从生产数据库复制数据。目标是可靠地获取所需数据，同时对源系统影响最小。

### 管理数据存储

数据需要一个存放的地方。数据工程师负责选择、实施和管理各种数据存储方案。这包括：

- 使用传统的**关系型数据库**（如PostgreSQL或MySQL）存储结构化数据。
- 使用**NoSQL数据库**（如MongoDB或Cassandra）处理更灵活的数据结构或高并发事务。
- 设置和维护针对分析查询优化的**数据仓库**（如Snowflake、BigQuery或Redshift）。
- 组织**数据湖**（通常使用Apache Hadoop HDFS或Amazon S3、Google Cloud Storage等云存储技术）来存储大量不同格式的原始数据。
  选择合适的存储系统取决于数据类型、访问方式、性能要求和成本等因素。我们将在第4章讨论存储选项。

### 数据清洗和转换

原始数据很少是完美的。它可能包含错误、不一致、缺失值，或者格式难以使用。数据工程师编写代码（通常使用SQL、Python或专业工具）来清洗、标准化和重塑数据，使其达到一致和可用的状态。这确保了数据分析师和数据科学家可以信任他们所使用的数据。这一步对于准确的报告和可靠的AI模型非常重要。

### 自动化和编排工作流

手动运行数据管道效率低下且容易出错。数据工程师使用工作流管理工具（如Apache Airflow或Prefect）来调度、自动化和监控数据管道。这确保了数据定期可靠地处理，并且任何故障都能被检测到并迅速解决。可以将这些工具视为数据管弦乐队的指挥，确保每个部分都在正确的时间运行。

### 监控、故障排除和优化

数据管道和存储系统需要持续关注。数据工程师监控系统性能、数据质量和管道执行情况。当出现问题时，比如管道失败、数据看起来不正确或系统运行缓慢，他们会调查根本原因并实施修复。他们还致力于优化管道和查询，使其运行更快并消耗更少资源，这在处理大型数据集时尤为重要。

### 基础设施管理

数据工程任务运行在计算基础设施上。这可能涉及管理服务器、使用云平台（如AWS、Google Cloud或Azure），以及配置数据处理和存储所需的软件。虽然有些组织有专门的基础设施团队，但数据工程师通常需要对底层系统有较好的了解。

### 团队协作

数据工程师并非孤立工作。他们与以下人员紧密合作：

- **数据分析师和数据科学家：** 了解他们的数据需求，并为他们提供所需干净、结构化的数据。
- **软件工程师：** 将数据收集机制集成到应用程序中。
- **业务相关人员：** 了解数据系统需要支持的目标。

下图展示了这些任务在典型数据流中是如何相互配合的：

> 数据从源头通过工程处理流向最终用户的简化视图。

这些任务共同确保组织的数据从其原始且常混乱的状态，转换为有价值的资产，从而驱动洞察和支持应用程序。随着您在课程中的学习，您将进一步了解如何有效执行这些活动所需的原理和工具。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  全面涵盖数据系统设计原则，包括存储、处理和可靠性，这些都是数据工程任务的基础。
- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF5G9MtxA8x-LEaZl3DQByNKhddiX2MJc941zxZtNwvLZQvF3tqVpMFFdN7hUqi1K4noKMCCSi-BAPvKUcWWAkX-1HTApCIVO89-8uFhQ_QagVIS4LYTL4Cl3rn7QXVCY06impLQLC4P5L3dgpsmXYQuOOEdaCKsmQg50pJ7qekJ2jE3Q==) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  一本现代数据工程指南，涵盖规划、构建和运行可靠的数据系统，包括数据管道、存储和质量管理。
- [Apache Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — The Apache Software Foundation (2024)
  Apache Airflow 的官方文档，这是一个广泛用于以编程方式编写、调度和监控数据管道的平台。
