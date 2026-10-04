---
course: "intro-data-engineering"
chapter: "what-is-data-engineering"
lesson: "data-lifecycle"
sourceId: 5493
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-1-what-is-data-engineering/data-lifecycle"
title: "数据生命周期"
description: "了解数据从生成到使用所经历的不同阶段。"
order: 4
plots: []
sourceHash: "dbf8002f4fbcbea018a639ef623d5619dad8f8f25be1a0f33dddfe966188be46"
sourceCorrections: []
---

就像工厂里的产品有生产过程一样，数据也从创建到最终使用或废弃遵循一条路径。理解这个过程，即**数据生命周期**，对数据工程师来说非常重要。它有助于弄清数据的来源、需要对其进行哪些操作，以及它最终如何发挥作用。可以把它看作是数据工程师用来构建可靠系统的路线图。

数据不会凭空出现并直接可供分析。它会经历几个不同的阶段：

1. **生成：** 这是数据诞生的地方。它可以来自众多来源：用户与网站或移动应用互动、传感器收集环境读数、交易系统记录销售、软件应用生成的日志，甚至是外部合作伙伴提供的数据流。在这个阶段，数据工程师较少参与创建，但需要理解他们最终将处理的数据的性质和来源。
2. **收集：** 数据生成后，需要进行收集。这可能包括从服务器读取文件、查询数据库、订阅流数据，或调用外部应用程序接口（API）。数据工程师经常设计和实现系统和流程，负责从各种来源可靠地收集这些原始数据。自动化在这里非常重要，以处理信息的持续流动。
3. **处理：** 原始数据很少以其原始形式有用。它通常包含错误、缺失值、不一致，或者需要重塑以适应分析或应用使用。处理阶段包括清洗、转换、验证、聚合和丰富数据。数据工程师构建数据管道，通常使用抽取、转换、加载（ETL）或抽取、加载、转换（ELT）模式（我们将在第3章介绍），来执行这些操作。这是数据工程师的主要职责，确保数据变得准确、一致并可供使用。
4. **存储：** 处理后的数据需要存放。根据其性质和使用方式，它可能存储在不同的系统中。例如，关系数据库用于结构化交易数据，数据仓库用于优化的分析查询，数据湖用于存储大量各种格式的原始或处理过的数据，或NoSQL数据库用于灵活的数据结构。数据工程师选择、设计和管理这些存储系统，确保数据高效且安全地存储。
5. **管理和治理：** 在整个生命周期中，数据需要得到妥善管理。这包括确保数据质量、实施安全措施来控制访问、遵守法规（如隐私法），以及维护元数据（关于数据的数据）。虽然专门的角色有时侧重于治理，但数据工程师通过在他们构建的管道和存储系统中实现控制来发挥作用。
6. **分析和使用：** 在此阶段，数据的价值得以实现。数据分析师、数据科学家、机器学习 (machine learning)工程师、商业智能工具和应用程序使用处理和存储的数据。他们可能构建报告、训练模型、获取洞察或驱动应用功能。数据工程师在这里的作用是确保数据对这些下游用户和系统来说易于访问、可靠且性能良好。
7. **归档/销毁：** 并非所有数据都需要永远保留，或至少不需要在活跃系统中保留。根据业务需求和合规要求，数据可能会被移至更便宜的长期归档存储或安全删除。数据工程师可以实施流程来自动化数据归档或删除策略。

以下图表展示了这些阶段的典型流程：

> 这是数据生命周期的简化视图，展示了从创建到使用的主要阶段，管理和治理始终都在。

理解这个生命周期使数据工程师能够预见需求，设计合适的系统，并确保数据从起源点流畅且可靠地流向可以产生洞察或驱动应用程序的地方。每个阶段都带来独特的挑战，并需要特定的工具和技术，我们将在本课程中学习这些。

## 参考资料

- [The Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media; Pages: 450
  对数据工程进行了全面介绍，将数据生命周期作为构建可靠数据系统的核心概念。
- [DAMA-DMBOK: Data Management Body of Knowledge](https://www.technicsdef.com/dmbok2-revised-edition/) — DAMA International (2017)
  Publisher: Technics Publications; Pages: 590
  行业数据管理实践标准参考，包含数据治理的原则和框架。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://books.google.com/books/about/Designing_Data_Intensive_Applications.html?id=oP3oDQAAQBAJ) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  对现代数据系统背后的技术和架构模式提供了深入理解，与数据生命周期的所有阶段相关。
