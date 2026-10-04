---
course: "intro-etl-pipelines"
chapter: "building-simple-etl-pipelines"
lesson: "what-is-etl-pipeline"
sourceId: 5579
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-5-building-simple-etl-pipelines/what-is-etl-pipeline"
title: "什么是 ETL 流水线？"
description: "将 ETL 流水线定义为一系列相互连接的 ETL 过程。"
order: 1
plots: []
sourceHash: "17c9a7571f86dec0254f8d83a852a8efa7fa05df2b89728233381f80b9cc8f9b"
sourceCorrections: []
---

数据处理通常涉及三个主要步骤：提取、转换和加载。提取数据是指从源头获取数据。转换数据涉及清理和重塑数据。加载数据是指将数据放入目标系统。当这些阶段被连接成一个自动化序列时，它们的真正效用便显现出来。这个连接起来的序列就是我们所说的 **ETL 流水线**。

可以将 ETL 流水线想象成一条为数据服务的自动化装配线。原始材料（您的初始数据）从一端进入，经过各种处理站（提取、转换），并在另一端以成品（干净、结构化且可供使用的数据）的形式出现，然后被存储起来（加载）。

因此，ETL 流水线是一系列按特定顺序执行的数据处理步骤：

1. **提取：** 从一个或多个源系统获取数据。
2. **转换：** 提取的数据根据预设规则进行清理、验证、结构化和丰富。
3. **加载：** 转换后的数据写入目标系统，例如数据库、数据仓库或数据湖。

流水线的显著特点是这些步骤是相互连接且通常是自动化的。提取阶段的输出作为转换阶段的输入，而转换的输出则成为加载阶段的输入。这就形成了从源头到目标的数据连续流动。

下面是一个说明此流程的简单图示：

> 一个典型的 ETL 流水线，显示了从数据源，经过提取、转换和加载阶段，到目标系统的顺序。

流水线被设计为可重复且可靠。无需在每次需要处理新数据时手动运行每个步骤，您可以定义一次流水线，然后它可以根据日程（例如：每天、每小时）自动执行，或由特定事件（例如：新文件到达）触发。

构建 ETL 流水线的主要目的是创建一致、自动化且可管理的数据移动和准备流程。这确保数据以正确的格式和质量抵达目标系统，可用于分析、报告或应用程序。

在本章的后续部分中，我们将更仔细地研究这些流水线的结构、用于构建和管理它们的工具类型，以及如何处理调度和监控等方面。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  讨论构建数据系统的基本原则和模式，为数据集成和 ETL 的设计与挑战提供背景。
- [What is ETL?](https://aws.amazon.com/what-is/etl/) — AWS (Amazon Web Services) (2023)
  从领先云提供商的角度清晰地解释了 ETL 管道、其组件和优势。
