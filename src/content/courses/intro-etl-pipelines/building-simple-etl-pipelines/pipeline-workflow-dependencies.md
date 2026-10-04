---
course: "intro-etl-pipelines"
chapter: "building-simple-etl-pipelines"
lesson: "pipeline-workflow-dependencies"
sourceId: 5580
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-5-building-simple-etl-pipelines/pipeline-workflow-dependencies"
title: "管道工作流与依赖关系"
description: "了解管道中的任务如何排序以及它们之间如何相互依赖。"
order: 2
plots: []
sourceHash: "9d5debbcecdc4c87517183fc3d3e9d2a257d5b718be2ca242b11558d371f3451"
sourceCorrections: []
---

抽取、转换和加载是数据处理中的基本操作阶段。虽然这些阶段可以被视为独立的操作，但它们真正的作用体现在将它们连接成一个自动化序列。在这个序列中，数据从一个阶段流向下一个阶段，从而构成了一个 ETL 管道。不要将其仅仅视为独立的 E、T 和 L 模块，而应将其看作一个旨在自动移动和整理数据的连接系统。

这些 E、T 和 L 任务（以及其中可能包含的更小的子任务）的排列和执行顺序定义了管道的**工作流**。工作流本质上是一张蓝图，它规定了管道中 *发生什么* 以及 *何时发生*。在最简单的情况下，工作流是线性的：首先抽取，然后转换，接着加载。

任何工作流的一个基本方面是**依赖关系**。依赖关系意味着一个任务必须成功完成后，另一个任务才能开始。想象一下，在你甚至还没从源头抽取数据之前就尝试将数据加载到数据仓库中，或者尝试对尚未到达的数据进行转换。这根本行不不通。

ETL 管道中最基本的依赖结构是顺序的：

1. **抽取：** 从源头获取数据。
2. **转换：** 清理、验证并重塑已抽取的数据。这一步*依赖于*抽取步骤的成功完成。
3. **加载：** 将已转换的数据插入到目标系统中。这一步*依赖于*转换步骤的成功完成。

我们可以图示这个简单的线性工作流：

> 一个简单的管道工作流，显示了顺序依赖关系。数据从抽取流向转换，然后从转换流向加载。

在此图中，箭头表示数据的流动和依赖关系。`Transform` 任务必须等到 `Extract` 完成才能开始，而 `Load` 任务必须等到 `Transform` 完成才能开始。

“尽管许多简单的管道遵循这种严格的线性路径，但工作流可以变得更复杂。设想一个场景：你从一个源头抽取客户数据，从另一个源头抽取订单数据。你可能会在之后的转换步骤中合并（连接）它们之前，对每个数据集*并行*执行一些初始的清理转换。”

让我们图示一个稍微复杂一些的工作流：

> 一个带有并行转换步骤的工作流。客户和订单数据在合并并加载之前，会独立地进行抽取和清理。

在此示例中：

- `Clean Customers` 仅依赖于 `Extract Customers`。
- `Clean Orders` 仅依赖于 `Extract Orders`。
- 这两个清理任务可以同时（并行）运行，因为它们彼此不依赖。
- 然而，`Join Customer & Order Data` 任务*同时*依赖于 `Clean Customers` 和 `Clean Orders` 的成功完成。
- 最后，`Load to Warehouse` 依赖于 `Join Data` 步骤。

理解这些工作流和依赖关系在多个方面都很重要：

- **正确性：** 它确保操作按逻辑顺序发生，从而产生期望的结果。你需要食材（抽取的数据）才能准备菜肴（转换），菜肴准备好后才能上菜（加载）。
- **错误处理：** 如果一个任务失败，理解依赖关系有助于判断哪些后续任务不应执行。如果 `Extract Customers` 失败，尝试 `Clean Customers` 或 `Join Data` 毫无意义。
- **效率：** 识别可以并行运行的任务（如上述 `Clean Customers` 和 `Clean Orders`）可以显著加快整个管道的执行时间。
- **故障排除：** 当管道失败时，从故障点反向追溯依赖关系通常是诊断根本原因的第一步。

当你开始设计管道时，即使是简单的管道，也始终要考虑操作的顺序以及每个步骤如何依赖于前一个步骤。规划出这个工作流并识别依赖关系是构建可靠且有效的 ETL 流程的基础一步。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball and Margy Ross (2013)
  Publisher: John Wiley & Sons, Inc.
  数据仓库和ETL的基础教材，系统地介绍了数据提取、转换和加载过程的设计，并讨论了它们之间的相互依赖关系。
- [Fundamentals of Data Engineering: Plan and Build Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108189/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  提供了构建数据管道的现代视角，详细阐述了ETL中工作流编排、任务依赖和并行执行的原理和方法。
- [Apache Airflow Documentation: Core Concepts](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/index.html) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  官方文档，解释了有向无环图（DAGs）、任务和操作符，它们是数据管道中工作流和依赖管理的实际实现。
