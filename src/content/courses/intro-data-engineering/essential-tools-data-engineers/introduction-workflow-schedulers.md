---
course: "intro-data-engineering"
chapter: "essential-tools-data-engineers"
lesson: "introduction-workflow-schedulers"
sourceId: 5601
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-6-essential-tools-data-engineers/introduction-workflow-schedulers"
title: "工作流调度器简介"
description: "了解用于自动化和调度数据工作流的工具。"
order: 5
plots: []
sourceHash: "a8eda59845e29015ab132dc5cf45cda9fcacfcc07a66d59486674e900c94b8d9"
sourceCorrections: []
---

在前面的章节中，特别是在讨论数据管道时，我们看到数据移动和转换通常涉及多个顺序步骤。例如，您可能需要从源系统提取数据、清洗数据、转换其结构，然后加载到数据仓库。想象一下每天甚至每小时手动执行这个过程。这会很快变得繁琐、耗时，并且容易出现人为错误。如果一个步骤失败了怎么办？如何确保只有在前一步成功后才运行下一步？这就是工作流调度器发挥作用的地方。

工作流调度器，也称为工作流编排器或工作流管理系统，是专门用于自动化、调度和监控任务序列的工具，我们称这些任务序列为工作流或数据管道。把它们想象成管弦乐队的指挥，确保每个乐器（或任务）在正确的时间以正确的顺序演奏其部分。

### 定义和可视化工作流

大多数工作流调度器允许您将工作流定义为一系列任务，并指定它们之间的依赖关系。表示这些关系的常用方式是使用有向无环图 (DAG)。

- **有向的：** 表示关系具有方向性；任务A必须在任务B开始之前完成。
- **无环的：** 表示工作流不包含循环；任务不能以形成循环的方式（不允许任务A -> 任务B -> 任务C -> 任务A）依赖于序列中稍后运行的任务。这确保工作流有明确的开始和结束。

通常，您使用代码（在常用工具中常为 Python）定义这些 DAG，这允许实现复杂的逻辑、动态任务生成以及与 Git 等版本控制系统集成。

这是一个表示工作流 DAG 的简单图表：

> 一个简单的工作流，数据被提取，由两个任务并行转换，然后加载。完成后会发送通知。

### 调度和执行

一旦工作流被定义，调度器就开始接管。其职责包括：

1. **调度：** 根据定义的时间表（例如，“每天凌晨 3:00 运行”、“每周二中午运行”）或外部事件（例如，“每当此存储位置出现新文件时运行”）触发工作流。
2. **管理依赖关系：** 确保任务仅在其所有上游依赖项成功完成后才开始。在上面的图表中，`加载到数据仓库` 仅在 `转换集合A` 和 `转换集合B` 都完成后运行。
3. **执行任务：** 运行与每个任务关联的实际代码或命令。这可能涉及运行 SQL 脚本、执行 Python 程序、与 API 交互或调用其他服务。
4. **监控：** 跟踪工作流和单个任务的状态（例如，运行中、成功、失败）。大多数调度器提供用户界面以查看进度。
5. **处理故障：** 当任务失败时，实现相应的处理逻辑。这可能包括自动重试任务一定次数，或者通过电子邮件或消息平台向数据工程师团队发送警报。

### 为什么调度器在数据工程中很重要？

使用工作流调度器相比手动执行或简单的 cron 作业具有多项优势：

- **自动化：** 减少人工干预的需求，节省时间和精力。
- **可靠性：** 系统地处理依赖、重试和故障，使管道运作有效。
- **可见性：** 提供一个中心位置来监控数据管道的状态和性能。日志通常集中管理，使调试更容易。
- **可扩展性：** 可以高效地管理包含许多任务和依赖的复杂工作流。
- **可维护性：** 将工作流定义为代码（通常称为“管道即代码”）使其更易于版本控制（使用 Git）、测试和协作。

### 工作流调度器示例

虽然本入门课程不会详细讲解，但了解一些广泛使用的工具名称会很有帮助：

- **Apache Airflow：** 一个非常流行的开源平台，使用 Python 定义 DAG。
- **Prefect：** 另一个开源选项，同样基于 Python，专注于现代数据技术栈的集成。
- **Dagster：** 一个开源工具，侧重于开发实践、测试和可观测性。
- **云服务提供商服务：** 主要的云平台，如 AWS（Step Functions, Managed Workflows for Apache Airflow）、Google Cloud（Cloud Composer, Workflows）和 Azure（Data Factory），提供托管式工作流编排服务，这些服务与其各自的其他云产品紧密结合。

这些工具为自动化您设计的数据管道提供了支持。它们协调各个部分，运行 SQL 查询（本章前面已介绍）、执行 Git 管理的脚本、与云存储和计算资源交互，以确保数据在您的系统中可靠地流动。随着您在数据工程方面的进展，理解和使用这些工具会变得越来越重要。

## 参考资料

- [Apache Airflow Documentation](https://airflow.apache.org/docs/) — Apache Software Foundation (2024)
  Publisher: Apache Software Foundation
  Apache Airflow 官方文档，详细介绍了如何使用 Python DAG 定义、调度和监控数据工作流。
- [Fundamentals of Data Engineering: Planning and Building Analytics Data Platforms](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108402/) — Joe Reis and Matt Housley (2022)
  Publisher: O'Reilly Media
  数据工程的综合指南，内容包括数据管道设计原则以及工作流编排在构建可靠数据平台中的作用。
- [AWS Step Functions Developer Guide](https://docs.aws.amazon.com/stepfunctions/latest/dg/welcome.html) — Amazon Web Services (AWS) (2025)
  Publisher: Amazon Web Services (AWS)
  解释了编排分布式应用和微服务的无服务器方法，展示了工作流概念在云环境中的实际应用。
- [Dagster Documentation](https://docs.dagster.io/) — Dagster Labs (2024)
  Dagster 的官方文档，侧重于以资产为中心的数据管道，并提供强大的开发、测试和可观察性支持。
