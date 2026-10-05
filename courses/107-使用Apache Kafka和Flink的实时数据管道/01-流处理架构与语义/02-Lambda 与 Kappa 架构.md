# Lambda 与 Kappa 架构

来源：[原文](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-1-stream-processing-architectures-semantics/lambda-vs-kappa-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建数据系统通常需要在延迟和吞吐量 (throughput)之间进行权衡。大数据早期，分布式系统难以提供低延迟更新，同时保证历史数据分析所需的高吞吐量、容错一致性，这带来了一些挑战。这一限制促进了特定架构模式的发展，这些模式旨在弥补现有工具的不足。随着 Apache Flink 等流处理引擎成熟并支持精确一次语义和有状态处理，这些模式也随之演变。了解从 Lambda 到 Kappa 架构的演进，对于设计能够尽量降低操作复杂程度同时不牺牲数据完整性的数据管道而言是必要的。

### Lambda 架构

Lambda 架构被引入以处理海量数据，它通过同时利用批处理和流处理方法。它采用多层方法来平衡延迟、吞吐量 (throughput)和容错能力。其核心理念是：批处理层通过定期处理整个主数据集，提供精确、全面的数据视图；而速度层则提供低延迟、近似的最新数据视图。

该架构由三个不同的组件组成：

1. **批处理层：** 这一层管理主数据集，它是一个不可变、只追加的原始数据集。它使用 Hadoop MapReduce 或 Apache Spark 等高延迟系统预计算批处理视图。这一层是数据源头。如果代码发生变化或出现错误，批处理层可以从头重新计算整个数据集以修正输出。
2. **速度层：** 这一层实时处理数据流。它通过只处理最新数据来弥补批处理层的高延迟。在过去，这一层使用 Apache Storm 等系统，这些系统常常为了速度而牺牲强一致性（采用至少一次语义）。
3. **服务层：** 这一层通过合并来自批处理视图和实时视图的结果来响应查询。

Lambda 架构的理论模型可以表示为总可查询视图 $V$ 的函数：

$V = V_{batch} \cup V_{realtime}$

这里，批处理视图 $V_{batch}$ 表示函数 $f$ 应用于时间点 $t_{batch}$ 的完整数据集 $D$，而 $V_{realtime}$ 则表示处理在 $t_{batch}$ 和当前时间 $t_{now}$ 之间到达的增量数据的结果。

> Lambda 架构将数据流分成“热”路径（红色）用于低延迟，以及“冷”路径（紫色）用于完整性。

虽然有效，但 Lambda 架构带来了显著的运维开销。“编码负担”是主要缺点：工程师必须为相同的业务逻辑维护两套不同的代码库，一套用于流处理框架，一套用于批处理框架。这种重复增加了逻辑分歧的可能性，即速度层由于实现差异而产生的结果与批处理层略有不同。此外，管理两套不同分布式系统的操作复杂程度，增加了 DevOps 和基础设施团队的负担。

### Kappa 架构

Kappa 架构通过将所有数据处理视为流处理问题来简化数据管道。它完全移除了批处理层，认为批处理只是一个有界流。这一转变得益于 Flink 等流处理引擎的演进，这些引擎支持精确一次语义、事件时间处理和可扩展的状态管理。

在 Kappa 架构中，不可变日志（Apache Kafka）作为规范数据存储。流处理作业处理实时数据和历史数据的再处理。当业务逻辑发生变化时，您无需运行单独的批处理作业。相反，您可以部署新版本的流式应用程序，并从 Kafka 日志的特定偏移量或从头开始重放数据。

该架构基于四个主要原则：

1. **一切皆流：** 批处理数据被视为一个流，只是它碰巧有一个已知的开始和结束。
2. **不可变日志作为数据源头：** 数据持久化在具有足够保留时间（通常使用压缩主题或分层存储）的分布式日志（Kafka）中。
3. **单一代码库：** 相同的应用程序代码处理实时事件和历史数据重放。
4. **通过重放进行再处理：** 为了修复错误或引入新功能，系统会重放输入流。

> Kappa 架构将数据管道统一为单一路径。重放历史数据包括回溯数据源上的消费者偏移量。

### 收敛与语义

从 Lambda 到 Kappa 的转变不仅仅是图表的简化。它代表着处理保证的转变。Lambda 架构假设速度层本质上不可靠或近似，因此需要批处理层在后期“修正”数据（最终一致性）。

现代流处理引擎通过检查点和状态管理来保证正确性。例如，Flink 的异步屏障快照使得系统即使在发生故障时也能保持一致状态。这一能力使得“修正”批处理层变得多余。

然而，实现 Kappa 架构需要解决**再处理吞吐量 (throughput)**的问题。当从 Kafka 重放数 TB 的历史数据时，流处理器必须以远高于实时到达速率的速度摄入数据。这通常需要：

1. **弹性伸缩：** 在重放期间临时增加 Flink 作业的并行度。
2. **背压处理：** 确保目标（数据库或下游 API）能够处理重放写入操作的激增。
3. **并行部署：** 通常不停止现有管道，而是启动新版本的管道来并行处理历史数据。一旦新管道追赶到流的实时前端，应用程序便会将流量切换到新版本。

### 比较复杂程度与延迟

在这两种架构之间做出选择时，需考虑团队的运维能力以及工作负载的要求。当“批处理”逻辑极其复杂且计算成本高昂时（例如，训练无法增量更新的大规模机器学习 (machine learning)模型），Lambda 架构通常受到青睐。Kappa 架构对于标准 ETL、数据分析和滚动聚合更优越，在这些场景中，代码可维护性和低延迟是优先考虑的。

以下图表显示了两种架构在不同维度上的相对权衡。



[交互图表：架构对比：Lambda 与 Kappa](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-1-stream-processing-architectures-semantics/lambda-vs-kappa-architecture#plot-qyn8tl)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "架构对比：Lambda 与 Kappa",
    "barmode": "group",
    "xaxis": {
      "title": "架构维度"
    },
    "yaxis": {
      "title": "得分（相对比例）"
    },
    "font": {
      "family": "Arial, sans-serif"
    },
    "margin": {
      "l": 50,
      "r": 50,
      "t": 50,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        "代码可维护性",
        "数据延迟",
        "操作复杂程度",
        "历史数据准确性"
      ],
      "y": [
        2,
        8,
        9,
        9
      ],
      "name": "Lambda",
      "type": "bar",
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "x": [
        "代码可维护性",
        "数据延迟",
        "操作复杂程度",
        "历史数据准确性"
      ],
      "y": [
        9,
        9,
        4,
        9
      ],
      "name": "Kappa",
      "type": "bar",
      "marker": {
        "color": "#69db7c"
      }
    }
  ]
}
```

</details>



> Kappa 架构显著提高了代码可维护性，降低了操作复杂程度，同时能达到经过良好调优的 Lambda 实现的准确性和延迟水平。

### 从批处理到流处理

在 Flink 生态系统中，批处理和流处理之间的区别由数据的有界性定义。批处理只是一个有界流。当您以“批处理模式”执行 Flink 作业时，会应用优化措施（例如排序输入数据而不是溢出到 RocksDB 状态），但 API 和底层逻辑保持一致。

这种统一使得工程师能够严格遵循 Kappa 理念进行开发。您可以使用 DataStream API 编写一次逻辑。如果需要回填数据，您可以将相同的逻辑应用于 Kafka 中有界范围的偏移量。这创建了一个确定性系统，其中：

$Result = f(EventLog_{0 \dots t})$

理解这种统一模型对于后续章节是必要的，我们将在这些章节中实现依赖这种确定性行为以确保精确一次处理的特定 Flink 模式。

## 参考资料

- [Big Data: Principles and Best Practices of Scalable Realtime Data Systems](https://www.manning.com/books/big-data) — Nathan Marz and James Warren (2015)
  Publisher: Manning Publications
  理解Lambda架构的权威资源，涵盖其设计动机、分层组件和容错数据处理模式。
- [Questioning the Lambda Architecture](https://www.oreilly.com/ideas/questioning-the-lambda-architecture) — Jay Kreps (2014)
  Journal: Confluent Blog; Publisher: Confluent
  介绍Kappa架构作为Lambda的替代方案，提倡以日志为中心的方法，将所有数据处理统一为流处理。
- [Apache Flink Documentation](https://flink.apache.org/docs/) — The Apache Flink Community (2025)
  提供关于Flink功能的全面信息，包括其统一的批流处理模型、精确一次语义和状态管理，这些都支持了Kappa架构。
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/) — The Apache Kafka Community (2024)
  详细说明Kafka作为分布式提交日志的角色、其保障措施和功能，这些使其成为像Kappa这样的现代数据架构中不可变的事实来源。
- [Stream Processing with Apache Flink: Fundamentals, StreamSQL, and Table API](https://learning.oreilly.com/library/view/stream-processing-with/9781491974284/) — Fabian Hueske and Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  一本使用Apache Flink构建实时数据管道的实用指南，涵盖处理语义、状态管理以及流优先设计的架构影响。

---

[上一节](01-%E5%88%86%E5%B8%83%E5%BC%8F%E6%97%A5%E5%BF%97%E7%9A%84%E5%8F%91%E5%B1%95.md) · [下一节](03-%E5%A4%84%E7%90%86%E4%BF%9D%E9%9A%9C%E5%92%8C%E5%90%AB%E4%B9%89.md)
