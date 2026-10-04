---
course: "intro-data-engineering"
chapter: "introduction-data-processing"
lesson: "understanding-compute-resources"
sourceId: 5581
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-5-introduction-data-processing/understanding-compute-resources"
title: "理解计算资源"
description: "了解数据处理任务所需的计算能力。"
order: 4
plots: []
sourceHash: "3c54a0b8f69f25df6d7cf65e67c6067f0c2c9d315cca57194a2493a296afbedc"
sourceCorrections: []
---

好的，我们已经讨论了数据是*如何*处理的，区分了按计划运行的批处理作业和持续处理数据的流处理。但实际执行处理的是什么呢？无论是大量批次还是实时流，转换原始数据都需要计算能力。可以把它想成车辆需要引擎才能移动。数据处理也需要它自己的“引擎”，我们称之为**计算资源**。

数据处理的核心包含计算、数据传输和临时存储。提供这种能力的主要组件是：

1. **CPU（中央处理器）：** 它是操作的“大脑”。CPU执行计算并运行处理逻辑中定义的指令（例如清理数据、应用转换或执行数学函数）。更复杂的转换或处理更大体量的数据通常需要更多的CPU能力。
2. **内存（RAM - 随机存取存储器）：** 可以把RAM想成计算机的短期工作区。正在处理的数据需要加载到内存中，以便CPU快速访问。处理大型数据集或执行需要同时保存许多中间结果的操作，都需要足够的RAM。如果RAM不足，处理速度会大幅下降，因为系统会开始使用较慢的磁盘存储作为临时内存（这个过程称为交换或分页）。

有时，特别是在机器学习 (machine learning)或复杂模拟（这些通常在数据工程步骤之后）中，另一个组件变得重要起来：

- **GPU（图形处理器）：** 尽管最初是为图形处理设计的，但GPU在执行某些类型的并行计算方面效率很高，速度非常快。对于一些专门的数据处理任务，特别是在人工智能和科学计算中，GPU可以比CPU提供显著的速度提升。对于一般的数据清理和转换，CPU通常是主要的“主力”。

### 根据任务调整资源

并非所有数据处理任务都需要同等对待。处理来自数千个设备的数TB传感器数据所需的资源，远多于分析一份小型日常销售报告。在数据工程中，这种根据任务需求调整处理能力的能力被称为**伸缩**。增加处理能力主要有两种方式：

1. **纵向扩展：** 这指使用一台*更强大*的机器。你可以增加CPU核心、添加更多RAM或在一台服务器上使用更快的存储。这就像将一辆普通汽车升级为一辆重型卡车。初期通常更简单，但可能会变得昂贵且有物理限制。
2. **横向扩展：** 这指将处理工作量分配到*多台*配合工作的机器上。这就像不用一辆巨型卡车，而是使用一队小型货车。这在大型数据处理框架中很常见（比如Apache Spark，我们稍后会提到）。它提供了更大的灵活性，并且通常可以处理比单台机器大得多的负载。

> 一个图表显示了两种处理工作负载的方法：纵向扩展使用一台强大的机器，而横向扩展使用多台配合工作的机器。

批处理通常能从横向扩展中获益，使得海量数据集能够在合理的时间范围内在多台机器上并行处理。流处理也可以通过横向扩展来处理高容量的入站数据，确保连续处理而没有延迟。

### 这些资源从何而来？

传统上，组织会购买并维护自己的物理服务器（称为**本地**基础设施）。如今，从Amazon Web Services (AWS)、Google Cloud Platform (GCP)或Microsoft Azure等**云服务商**租用计算资源变得越来越普遍。

云平台提供了显著的灵活性：

- **弹性：** 你可以在需要时（例如，处理大型批处理作业）轻松请求更多资源，并在之后将其退还，通常只为你使用的部分付费。
- **多样性：** 它们提供了针对不同任务优化的各种机器类型（例如，高CPU型、高内存型、配备GPU型）。

我们将在第6章更详细地查看云平台。目前，请理解数据处理需要基础的计算资源（CPU、RAM），并且你需要有方法根据你的特定任务适当地伸缩这些资源，无论是使用自己的硬件还是借助云服务。有效管理这些资源是数据工程的核心部分，它能确保数据处理的可靠性和成本效益。

## 参考资料

- [Computer Organization and Design RISC-V Edition: The Hardware/Software Interface](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6) — David A. Patterson and John L. Hennessy (2021)
  Publisher: Morgan Kaufmann; Pages: 736; DOI: [10.1016/C2019-0-02555-5](https://doi.org/10.1016/C2019-0-02555-5)
  这本经典教材全面阐述了CPU、内存和计算机体系结构的基础知识，对于理解核心计算资源至关重要。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  这本备受推崇的书深入解释了扩展策略、分布式系统和数据处理的架构原理，与横向扩展和高效资源利用直接相关。
- [Overview of Amazon Web Services](https://docs.aws.amazon.com/whitepapers/latest/aws-overview/aws-overview.pdf) — AWS (2025)
  Publisher: Amazon Web Services
  这份官方白皮书介绍了云计算的基本概念，包括弹性、可扩展性以及各种服务模型的优势，这对于理解现代数据处理资源的配置和管理至关重要。
