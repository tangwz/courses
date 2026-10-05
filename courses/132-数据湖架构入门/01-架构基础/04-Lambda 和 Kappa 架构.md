# Lambda 和 Kappa 架构

来源：[原文](https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-1-architectural-foundations/lambda-and-kappa-architectures)

[返回章节目录](README.md) · [返回课程目录](../README.md)

设计数据湖常常需要权衡两个相互冲突的要求：对低延迟更新的需要，以及对全面、准确历史数据分析的需要。虽然 Medallion 架构为数据质量（Bronze、Silver、Gold）提供了一种逻辑上的组织方式，但它没有明确规定数据在这些阶段流动的时机或机制。这时，数据处理架构就发挥了作用。

构建数据处理管道的两种主要模式是 Lambda 架构和 Kappa 架构。这些模式定义了批处理和流处理如何协作以提供数据视图。

### Lambda 架构

Lambda 架构旨在通过结合批处理和流处理方法来处理海量数据。它采用混合方法，试图平衡延迟、吞吐量 (throughput)和容错性。

Lambda 架构的核心理念基于以下公式：

$Query = f(所有数据)$

然而，实时计算“所有数据”上的函数，其计算开销大，且通常不可行。为此，Lambda 将工作负载分为三个不同的层：

1. **批处理层：** 此层管理主数据集（一个不可变、只追加的原始数据集），并预计算批处理视图。它偏重一致性和完整性，而非速度。在数据湖的背景下，这通常是存储在 S3 或 Azure Blob 上 Parquet 或 Avro 格式的历史数据。
2. **速度层：** 此层只处理最新数据。它通过提供最新事件的实时视图来弥补批处理层的高延迟。它偏重低延迟，而非绝对精度或完整性。
3. **服务层：** 此层为批处理视图创建索引，以便进行查询。当查询到达时，系统会合并批处理视图和实时视图的结果，以提供完整答案。

> 数据进入系统后分成两条路径。热路径（速度层）提供即时结果，而冷路径（批处理层）确保长期准确性和纠正。

#### 优点与缺点

Lambda 架构的主要优势是容错性。如果速度层因 bug 或迟到数据产生了错误结果，批处理层最终会在下一周期用正确、核对过的数据覆盖它们。这提供了“自愈”能力。

然而，运营成本较高。您实际上维护着两套独立的 codebase：一套用于流处理系统（例如 Apache Flink 或 Spark Streaming），另一套用于批处理系统（例如标准的 Apache Spark 或 dbt）。在这两种不同的处理方法之间保持业务逻辑同步，是工程中常见的错误源。

### Kappa 架构

Kappa 架构的出现是对 Lambda 架构中维护两条并行管道所带来的繁杂性的回应。它提出，如果您的流处理系统足够强大，就不需要独立的批处理层。

在 Kappa 架构中，一切皆流。批处理被简单地视为一个具有有限数据集（起始点和结束点）的流处理任务，而实时处理则是一个具有无限数据集的流任务。

该架构由两个主要组成部分构成：

1. **不可变日志：** 记录系统是一个分布式日志（如 Apache Kafka 或 Amazon Kinesis），它能存储数据较长时间。
2. **流处理引擎：** 单一引擎处理日志中的数据以生成服务数据库视图。

如果您需要重新计算数据（这是 Lambda 架构中批处理层处理的要求），您只需使用相同的代码逻辑从日志开头重放流，即可有效地重新处理历史数据。

> 统一管道处理实时数据摄取和历史数据再处理。重放通过重置分布式日志中的偏移量来实现。

#### 优点与缺点

Kappa 通过统一代码库，大幅简化了基础设施。开发者只需编写一次转换逻辑。然而，它在数据保留方面引入了特定的难题。因为“批处理”功能依赖于重放流，所以底层日志存储必须能够保留可能达到 PB 级的历史数据，或者您必须实施一种分层策略，将较旧的日志段卸载到对象存储（数据湖），但仍保持可重放。

### Lambda 与 Kappa 的选择

选择合适的模式应考虑您的延迟要求和转换的繁杂程度。

Lambda 架构在以下情况下仍具有适用性：

- **算法差异：** 用于实时近似的算法与用于批处理准确性的算法（例如，机器学习 (machine learning)推理 (inference)与模型训练）明显不同。
- **频繁再处理：** 您经常需要在整个 PB 级数据集上重新运行历史数据，而通过流式引擎重放可能太慢。

Kappa 架构越来越受现代数据平台青睐，原因在于：

- **逻辑一致性：** 它消除了批处理层和速度层之间逻辑漂移的风险。
- **工具成熟度：** 现代流式引擎，如 Apache Flink 和 Spark Structured Streaming，能有效处理有状态处理和迟到数据，从而减少了对纠正性批处理层的需求。

### 演进：“Kappa Plus”数据湖

现代开放表格式（如 Delta Lake 和 Apache Iceberg）促成了一种变体，通常被称为“Kappa Plus”或“统一架构”。

在这种模式下，数据湖本身充当流式接收器和源。由于这些表格式支持 ACID 事务和高效的 upsert，您可以直接将数据流式传输到您的 Bronze 表中。下游的 Silver 和 Gold 表可以以微批或连续流的方式处理这些数据。这使得对象存储（S3/ADLS）能够充当“无限保留日志”，解决了 Kafka 等系统的保留限制，同时保持了 Kappa 的单一管道简洁性。

通过将计算与存储分离，并使用事务性表格式，您可以获得流的低延迟和批处理层的可伸缩性，有效地结合了两种架构的优点。

## 参考资料

- [Big Data: Principles and Best Practices of Scalable Realtime Data Systems](https://www.manning.com/books/big-data) — Nathan Marz and James Warren (2015)
  Publisher: Manning Publications
  这本基础书籍介绍并全面阐述了Lambda架构。
- [Questioning the Lambda Architecture](https://www.oreilly.com/radar/questioning-the-lambda-architecture/) — Jay Kreps (2014)
  Publisher: O'Reilly
  这篇有影响力的文章批判性地审视了Lambda架构，并提出了Kappa架构作为一种简化的替代方案。
- [The Log: What every software engineer should know about real-time data's unifying abstraction](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGywmjcfl5lnvuK8BiO4hC6jlszAZykatFO5PvsKG0eCcf__a9TIXG1exzPnO0qEwo1HM0Dp-f2BkFyw7do0rLR_CYzFTwxBnWNfJcAJKCmYVvT05DAkAxtAx7Hb9owWqQGxeGGxVQoGH3WBgTzTnOdnS1Xv9U0RkK5FjW_h427rR1P6oYjG5K6y62iG1G3k0X70hS57kG8uC2p2r2w2c4f1d3e1a0b3f) — Jay Kreps (2013)
  Publisher: Confluent Blog
  讨论了分布式日志作为实时数据系统的统一概念，构成了Kappa架构的理论基础。
- [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) — Apache Spark Development Team (2023)
  Publisher: Apache Software Foundation
  Apache Spark Structured Streaming的官方文档，展示了单个API如何处理批处理和流处理，支持Kappa方法。
- [Delta Lake Documentation](https://docs.delta.io/latest/index.html) — The Delta Lake Project (2023)
  Publisher: The Delta Lake Project
  Delta Lake的官方文档，详细介绍了其ACID事务和更新插入等功能，实现了数据湖中的“Kappa Plus”架构。

---

[上一节](03-%E5%A5%96%E7%AB%A0%E6%9E%B6%E6%9E%84.md) · [下一节](../02-%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F%E4%B8%8E%E4%BC%98%E5%8C%96/01-%E8%A1%8C%E5%BC%8F%E5%AD%98%E5%82%A8%E4%B8%8E%E5%88%97%E5%BC%8F%E5%AD%98%E5%82%A8.md)
