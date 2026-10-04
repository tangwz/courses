# ETL 介绍：抽取、转换、加载

来源：[原文](https://apxml.com/zh/courses/intro-etl-pipelines/chapter-1-understanding-etl-fundamentals/introducing-etl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据集成通常涉及从不同来源收集信息并将其整合以协同使用。实现这一目标最成熟且广泛使用的方法之一是 **ETL**，它代表着**抽取（Extract）、转换（Transform）、加载（Load）**。ETL 是一个系统的三步流程，用于将数据从其来源移动到可以由应用程序分析或使用的目标位置。

让我们详细说明每个阶段具体发生了什么：

### 抽取（E）

第一步是**抽取（Extract）**。这涉及从一个或多个源系统读取和获取数据。这些数据来源可以是数据存在的任何地方：

- 关系型数据库（如 PostgreSQL、MySQL、SQL Server）
- 文件（如 CSV、JSON、XML、日志）
- 来自网络服务的 API（应用程序编程接口）
- NoSQL 数据库
- 遗留系统

抽取步骤的主要目的仅仅是将所需数据从其原始位置“取出”。在此阶段，数据通常处于原始、未修改的格式。我们识别所需特定数据，并将其拉取到处理区域（有时称为暂存区），为下一步做好准备。

### 转换（T）

数据抽取完成后，\*\*转换（Transform）\*\*步骤便开始。这通常是整个流程中最复杂的部分。在此处，原始数据被清洗、验证和重塑，以满足目标系统的要求和预期用途（如分析或报告）。常见的转换活动包括：

- **清洗：** 处理错误、不一致或缺失值（例如，用默认值填充空白字段，纠正拼写错误）。
- **标准化：** 确保数据符合特定格式（例如，将所有日期格式转换为 `YYYY-MM-DD`，确保州缩写一致）。
- **验证：** 检查数据是否符合特定规则（例如，确保电子邮件地址字段确实包含有效的电子邮件格式）。
- **丰富化：** 添加新信息，可能通过合并来自不同来源的数据或计算新字段（例如，根据数量和单价计算总价，根据邮政编码添加人口统计信息）。
- **结构化：** 重塑数据，可能通过连接表、拆分列或数据透视。
- **聚合：** 汇总数据（例如，计算每个区域的总销售额）。

转换阶段确保数据变得一致、准确，并适合其最终目标。

### 加载（L）

最后一步是**加载（Load）**。数据转换完成后，需要将其写入目标系统。此目标系统通常是数据库、数据仓库、数据湖或为分析或操作用途设计的其他系统。

加载可以通过不同方式进行：

- **全量加载：** 清除目标表中现有数据，并将其完全替换为新处理过的数据。这更简单，但对于大型数据集可能较慢。
- **增量加载：** 仅添加新记录或更新自上次加载以来发生更改的现有记录。这对于持续更新更高效。

加载步骤使准备好的数据可供最终用户、分析师、数据科学家或需要它的应用程序使用。

### ETL 流程

从视觉上看，此流程形成一个管道，数据按顺序流经这三个阶段：

> 一张图表说明了数据如何从源系统通过抽取、转换和加载阶段流入目标系统。

总而言之，ETL 是数据管理中的一个基本流程。它提供了一种结构化的方法来：

1. 从各种来源**抽取**数据。
2. 将其**转换**为可用、一致的格式。
3. 将其**加载**到目标系统进行分析或其它目的。

理解这三个不同的阶段是设计和构建高效数据管道的第一步。在接下来的章节中，我们将更详细地研究每个阶段。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball and Margy Ross (2013)
  Publisher: John Wiley & Sons
  本书是数据仓库的奠基性资源，全面介绍了 ETL 流程、维度建模以及数据集成中的实践方法。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  尽管本书并非仅关注 ETL，但它提供了关于数据集成、数据处理和分布式系统的现代视角，为理解 ETL 模式在当代数据架构中的重要性提供了有力的背景认知。
- [Fundamentals of Data Engineering: Plan and Build Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis and Matt Housley (2022)
  Publisher: O'Reilly Media
  本书提供了数据工程实践的当代视角，包括对数据摄取、转换和加载策略的详细讨论，对于理解现代 ETL 概念具有高度相关性。

---

[上一节](01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%95%B0%E6%8D%AE%E6%95%B4%E5%90%88%EF%BC%9F.md) · [下一节](03-ETL%E6%B5%81%E7%A8%8B%E7%9A%84%E7%9B%AE%E7%9A%84.md)
