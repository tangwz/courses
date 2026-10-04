---
course: "intro-etl-pipelines"
chapter: "building-simple-etl-pipelines"
lesson: "etl-tools-categories"
sourceId: 5582
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-5-building-simple-etl-pipelines/etl-tools-categories"
title: "ETL 工具分类介绍"
description: "获取构建ETL管道所用不同类型工具（例如，基于GUI、基于代码）的概览。"
order: 3
plots: []
sourceHash: "91786c7cecc072bd7d1f05993bbae2a7735d1b517d3f7036e04424a9f4b82bdc"
sourceCorrections: []
---

ETL管道是将提取、转换和加载的各个阶段整合到自动化工作流程中。每次数据更新都手动运行每个步骤是不实际的，尤其是随着数据量和处理复杂性的增加。ETL工具在此变得必不可少。它们提供了定义、执行、调度和管理这些管道的框架和界面。

可以把构建ETL管道想象成建造一个复杂的项目。你可以尝试用基本材料从头开始构建一切，或者使用专为特定任务设计的专业电动工具。ETL工具就是用于数据集成工作的专业电动工具。它们处理许多底层的技术细节，让你能专注于数据 *如何* 被处理的逻辑。

尽管有大量的特定工具可用，它们通常分为几大类，基于你与它们交互的方式以及它们运行的位置。了解这些类别有助于为你的需求选择合适的方法。

### ETL 工具分类

选择ETL工具通常取决于团队的技术技能、数据转换的复杂程度、预算以及现有的技术基础设施。让我们看看常见的类型：

1. **可视化（基于图形用户界面）工具：**

   - **工作方式：** 这些工具提供图形用户界面（GUI），你通常将代表数据源、转换和目标的组件拖放到画布上，并连接它们来定义数据流。你通过菜单和表单配置每个组件，而不是编写大量代码。
   - **交互方式：** 主要通过视觉操作，点击选择。
   - **优点：** 对于初学者或不习惯编程的人来说通常更容易学习。适合流程的可视化。可以加快标准任务的开发速度。
   - **缺点：** 与编写代码相比，对于高度定制或复杂的转换可能提供较少的灵活性。有时可能是专有软件，并可能导致厂商绑定。
   - **例子：** Apache NiFi、Talend Open Studio（可视化部分）、Informatica PowerCenter等工具，以及许多基于云的服务都提供强大的可视化界面。
2. **脚本/代码类工具与库：**

   - **工作方式：** 这种方法涉及使用Python、Java或Scala等编程语言编写代码，并使用专门用于数据处理和工作流编排的库。你直接在代码中定义提取、转换和加载的逻辑。
   - **交互方式：** 编写和执行脚本或程序。
   - **优点：** 对管道的各个方面提供最大的灵活性和控制。可以处理高度复杂和定制的逻辑。运用已有的编程技能。通常使用开源库，避免厂商绑定。
   - **缺点：** 需要编程知识。对于简单任务，初期开发时间可能比可视化工具长。需要仔细的代码管理和测试。
   - **例子：** 使用Python及Pandas（用于转换）、SQLAlchemy（用于数据库交互）等库，以及Apache Airflow或Prefect等工作流编排工具。
3. **云平台集成服务：**

   - **工作方式：** 主要云提供商（如AWS、Google Cloud、Microsoft Azure）提供一套专门用于数据集成和ETL的托管服务。这些服务旨在在其各自的云生态系统中运行。它们通常结合了可视化界面和加入定制代码的能力。
   - **交互方式：** 通常通过云提供商的网页控制台、命令行工具或API。通常将可视化配置与代码选项结合。
   - **优点：** 与其他云服务（存储、数据库、机器学习 (machine learning)）紧密集成。可伸缩性和托管基础设施（提供商负责服务器、补丁等）。通常提供按使用量付费的定价模式。
   - **缺点：** 可能导致特定云生态系统内的厂商绑定。成本可能随使用量增加。初期可能需要较陡峭的学习曲线来熟悉具体服务。
   - **例子：** AWS Glue、Azure Data Factory、Google Cloud Dataflow/Dataproc。
4. **企业级数据集成平台：**

   - **工作方式：** 这些通常是全面的商业软件套件，提供广泛的功能，包括数据质量管理、元数据管理、主数据管理和高级治理能力。它们通常结合了可视化开发、大量连接器和企业级功能。
   - **交互方式：** 通常是可视化界面、配置控制台以及可能的脚本界面的组合。
   - **优点：** 为数据需求复杂的组织提供非常广泛的功能集。提供强大的支持、治理和安全功能。
   - **缺点：** 成本可能非常高。可能需要专业技能才能有效管理和使用。对于简单的ETL任务来说可能过于复杂。
   - **例子：** Informatica PowerCenter、IBM InfoSphere DataStage、SAP Data Services。

> 一张图表，展示了ETL工具的主要分类及其各自的主要特点。

对于本入门课程，我们将侧重于那些无论最终使用哪种具体工具类别都适用的基本原理。尽管可视化工具提供了设计管道的图形方式，基于代码的方法提供了高度灵活性，但数据提取、应用转换和加载结果的核心原理保持不变。了解这些类别会给你提供一个方向，助你开始构建自己的简单管道。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108424/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  提供了对数据工程原则的现代视角，涵盖了现代数据管道的设计和实现，包括ETL工具的各种方法。
- [Cloud Data Ecosystems: How to Build a Modern Data Stack for Your Business](https://link.springer.com/book/10.1007/978-1-4842-7910-8) — Bill Schmarzo, Kevin S. K. Lee (2022)
  Publisher: Apress; DOI: [10.1007/978-1-4842-7910-8](https://doi.org/10.1007/978-1-4842-7910-8)
  讨论了现代云数据堆栈的架构和组成部分，包括云服务提供商提供的各种数据集成服务和工具。
- [Data Pipelines with Apache Airflow](https://www.manning.com/books/data-pipelines-with-apache-airflow) — Bas P. Harenslak, Julian Rutger de Ruiter (2021)
  Publisher: Manning Publications
  一本使用Apache Airflow构建和管理数据管道的实用指南，展示了用于基于代码的ETL编排的关键开源工具。
