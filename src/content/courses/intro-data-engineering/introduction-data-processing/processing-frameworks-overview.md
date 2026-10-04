---
course: "intro-data-engineering"
chapter: "introduction-data-processing"
lesson: "processing-frameworks-overview"
sourceId: 5578
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-5-introduction-data-processing/processing-frameworks-overview"
title: "处理框架概述"
description: "对数据处理工具（如Spark思路）进行概括性介绍。"
order: 3
plots: []
sourceHash: "d059d3231e8e13f3fdaca98f98c5c47803d06f00ea0a9a598337217c02abb692"
sourceCorrections: []
---

您现在已明白数据需要处理，无论是定期批量还是持续流式。但这种处理究竟如何发生，特别是在处理可能的海量数据时？手动管理多台机器上的计算、处理故障并确保效率将极其复杂。这正是数据处理框架发挥作用的地方。

您可以把处理框架看作一种专门的软件系统，旨在简化和管理大规模数据处理。数据工程师无需编写低层代码来分发计算和处理错误，而是使用这些框架，它们提供更高层次的工具和抽象，以可靠高效地完成工作。

### 为何使用处理框架？

框架提供多项显著优点：

1. **可伸缩性：** 它们被设计用于在多台计算机（通常称为集群）上分发处理任务。如果数据量增加，通常只需向集群添加更多机器，框架便会负责分发工作。这使得系统能够横向扩展，以应对不断增长的数据量。
2. **容错性：** 处理大型数据集可能需要时间，且硬件故障可能发生。框架被设计用于检测故障（如机器崩溃），并自动重试任务或将其重新调度到正常工作的机器上，确保整个任务成功完成。
3. **抽象性：** 框架提供编程接口（API），让工程师能够定义需要进行 *何种* 处理（例如，过滤数据、聚合值），而无需担忧数据在集群中 *如何* 分布或故障如何管理等复杂细节。
4. **效率：** 许多框架包含复杂的优化器，能够找出执行您定义的处理步骤最有效的方法。它们处理数据本地性等方面（在数据存储位置附近处理数据），以最大程度地减少网络流量。

### 常见框架一览

尽管存在许多处理框架，这里有几个您可能会听到的基本范例：

- **Apache Hadoop MapReduce：** 这是用于大规模批处理的最初框架之一。它将任务分解为两个主要阶段：一个 *Map* 阶段，数据在不同节点上并行处理；以及一个 *Reduce* 阶段，Map 阶段的结果被聚合或汇总。尽管现在通常更倾向于使用速度更快、更灵活的新框架，MapReduce 模式为分布式数据处理奠定了基本原则。
- **Apache Spark：** Spark 是一个广泛使用的通用分布式处理框架。它受欢迎的原因是它能比 Hadoop MapReduce 更快地执行计算，主要通过将中间数据保存在内存中，而不是不断写回磁盘。Spark 用途广泛；它支持批处理、流处理（通过 Spark Streaming 或 Structured Streaming）、机器学习 (machine learning)（`MLlib`）和图处理（`GraphX`）。它提供 Scala、Java、Python（`PySpark`）和 R 等语言的 API，让许多开发人员都能使用。
- **流处理专用框架：** 对于需要极低延迟实时处理的应用，**Apache Flink** 和 **Apache Kafka Streams** 等框架表现突出。它们经过专门优化，用于处理持续到达的数据事件流。

> 一个图表，显示处理框架如何协调多个工作机器上的任务。

### 框架与计算资源

如前所述，这些框架并非独立运行。它们在计算机集群上运行，这些集群提供必要的 CPU、内存和存储（计算资源）。框架充当“大脑”，管理任务和数据如何分配并在这些资源上执行。AWS、Google Cloud 和 Azure 等云平台为 Spark 和 Flink 等常用框架提供托管服务，这使得配置和管理底层计算基础设施变得更加简便。

总之，数据处理框架是数据工程师工具箱中的重要工具。它们提供必要的结构、自动化和韧性来处理大量数据，从而将原始信息转化为有价值的见解，无论是通过周期性批量任务还是持续实时分析。明白这些工具的存在以及它们解决的问题，是学习数据工程的一个基本步骤。

## 参考资料

- [MapReduce: Simplified Data Processing on Large Clusters](https://www.usenix.org/legacy/events/osdi04/tech/full_papers/dean/dean.pdf) — Jeffrey Dean and Sanjay Ghemawat (2004)
  Journal: OSDI '04: 6th Symposium on Operating System Design and Implementation; Publisher: USENIX Association; Pages: 137-150; DOI: [10.1145/1070503.1070546](https://doi.org/10.1145/1070503.1070546)
  介绍MapReduce编程模型在大规模数据处理中应用的奠基性论文。
- [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf) — Matei Zaharia, Mosharaf Chowdhury, Michael Franklin, Scott Shenker, and Ion Stoica (2010)
  Journal: HotCloud '10: 2nd USENIX Workshop on Hot Topics in Cloud Computing; Publisher: USENIX; Pages: 1-10; DOI: [10.1145/1863103.1863113](https://doi.org/10.1145/1863103.1863113)
  介绍Apache Spark的论文，强调其内存处理能力以实现更快速的数据分析。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  全面概述分布式系统，包含可伸缩性、容错性和各种处理模型（批处理和流处理）等主题。
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/) — The Apache Software Foundation (Current)
  Publisher: The Apache Software Foundation
  Apache Spark的官方文档，详细介绍其架构、API以及各种处理需求的用例。
- [Apache Flink Documentation](https://flink.apache.org/docs/latest/) — The Apache Software Foundation (2025)
  Publisher: The Apache Software Foundation
  Apache Flink的官方文档，提供其流处理能力和实时数据分析的见解。
