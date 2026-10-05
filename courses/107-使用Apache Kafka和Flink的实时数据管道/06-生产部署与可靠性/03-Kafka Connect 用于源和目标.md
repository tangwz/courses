# Kafka Connect 用于源和目标

来源：[原文](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/kafka-connect-sink-source)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建可靠的数据摄取层通常会促使工程师为 PostgreSQL、Elasticsearch 或 S3 等标准数据系统编写自定义生产者和消费者。这种做法会带来大量的技术债务。自定义集成代码需要持续维护，以处理网络分区、模式变更和背压机制。Apache Kafka Connect 通过提供一个可扩展的声明式框架来解决这个问题，用于在 Kafka 和其他系统之间移动数据。您无需编写代码，而是定义配置 JSON 来指定数据流向，而 Connect 框架则负责分发、容错和偏移量管理。

### 分布式执行模型

在生产环境部署时，Kafka Connect 应以**分布式模式**运行。与独立模式不同，独立模式在单个进程中运行连接器，缺少高可用性，分布式模式将工作分散到工作节点集群中。这些节点通过 Kafka 组协调协议组成一个组，类似于消费者组。

运行时层级包含三个组成部分：

1. **工作节点：** 执行 Connect 框架的 JVM 进程。它们在数据处理方面是无状态的，但在集群成员身份方面是有状态的。
2. **连接器：** 任务的逻辑定义，管理将数据复制分解为多个任务。
3. **任务：** 实际执行单元。源任务从外部系统拉取数据并写入 Kafka；目标任务从 Kafka 拉取数据并写入外部系统。

当您将连接器配置提交到 REST API 时，工作节点会选举一个领导者来确定任务分配。如果一个工作节点出现故障，集群会检测到心跳丢失并触发重新平衡，将失败工作节点的任务重新分配给其余活跃节点。

> 分布式 Kafka Connect 集群的架构，展示了工作节点、任务以及用于协调的内部状态主题之间的关系。

### 源连接器和变更数据捕获

源连接器用于导入数据。尽管简单的连接器可能会使用 `SELECT * FROM table WHERE updated_at > ?` 轮询数据库，但这种基于查询的方法经常会遗漏硬删除，并对数据库执行引擎造成较大负担。

对于高保真数据管道，\*\*变更数据捕获（CDC）\*\*是更受推崇的模式。CDC 连接器（例如 Debezium 提供的）作为数据库事务日志（如 PostgreSQL WAL 或 MySQL binlog）的客户端。它们以事件形式捕获所有行级变更，包括插入、更新和删除。

源连接器的状态管理依赖于**源偏移量**。连接器会定期将其位置（例如日志序列号）保存到 `connect-offsets` 主题中。重启后，连接器会读取此主题，从上次中断的地方继续处理。

为优化源连接器的吞吐量 (throughput)，您必须使任务数量与源系统的并行处理能力相匹配。对于数据库，并行处理能力通常受限于分片或分区的数量。

### 目标连接器和交付语义

目标连接器将数据从 Kafka 主题导出到外部系统。这里最主要的工程挑战是确保数据一致性。Kafka Connect 保证**至少一次**交付。在故障场景中，例如工作节点在写入数据库后但在向 Kafka 提交偏移量之前崩溃，消息将被重新交付和重新处理。

为实现**精确一次**语义（或有效精确一次），目标系统必须支持幂等性。幂等的目标操作无论应用多少次，都会产生相同的结果。

例如，在写入键值存储或关系型数据库时，您应使用“upsert”（更新或插入）语义，而不是简单的追加操作。如果目标系统是 S3 等对象存储，由于最终一致性，直接实现精确一次会更困难，但可以通过基于 Kafka 偏移量确定性地对文件进行分区来管理。

批处理是另一个优化方向。目标任务会从 Kafka 累积记录，直到达到大小阈值或时间限制。在高延迟目标（如 S3 或 Snowflake）中，批处理大小 ($B$) 和吞吐量 (throughput) ($T$) 之间的关系通常遵循收益递减的对数曲线。

$T_{吞吐量} \approx \frac{B_{批处理}}{L_{网络} + T_{处理}(B_{批处理})}$

其中 $L_{网络}$ 是往返延迟，$T_{处理}$ 是目标系统摄取该批次所需的时间。



[交互图表：目标吞吐量与批处理大小配置](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/kafka-connect-sink-source#plot-8kxst8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "目标吞吐量与批处理大小配置",
    "xaxis": {
      "title": "批处理大小（记录数）",
      "showgrid": true,
      "color": "#495057"
    },
    "yaxis": {
      "title": "吞吐量（记录数/秒）",
      "showgrid": true,
      "color": "#495057"
    },
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "#ffffff",
    "font": {
      "family": "Helvetica",
      "color": "#343a40"
    },
    "margin": {
      "l": 60,
      "r": 30,
      "t": 50,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        1,
        10,
        50,
        100,
        500,
        1000,
        2000,
        5000
      ],
      "y": [
        15,
        120,
        450,
        800,
        2100,
        2800,
        3100,
        3150
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "line": {
        "color": "#339af0",
        "width": 3
      },
      "marker": {
        "color": "#1c7ed6",
        "size": 8
      }
    }
  ]
}
```

</details>



> `consumer.max.poll.records` 和连接器特定批处理大小对摄取吞吐量的影响。小批次会产生高网络开销，而超大批次则会触及处理限制。

### 单消息转换 (SMT)

Kafka Connect 允许通过单消息转换 (SMT) 进行轻量级数据操作。SMTs 对通过 Connect 框架的消息进行操作，无论是在写入 Kafka（源）之前，还是在发送到目标（Sink）之前。

SMTs 的常见用途包含：

- **掩码：** 在个人身份信息 (PII) 进入 Broker 之前进行混淆处理。
- **路由：** 根据字段内容更改目标主题。
- **扁平化：** 将复杂的嵌套结构展平为扁平层级，以适应需要表格数据的系统。

但是，SMTs 适用于简单的无状态转换。不要将 SMTs 用于复杂的聚合、流连接或繁重计算。这些任务属于您架构中的 Flink 层。SMTs 在 Connect 工作节点的 IO 线程中同步运行；在此处进行繁重处理会阻塞管道并增加消费者滞后。

### 与 Flink 集成

在健壮的 Kappa 架构中，Kafka Connect 和 Flink 扮演着互补的角色。Kafka Connect 负责“最后一公里”集成，将原始数据摄入 Kafka (EL) 并将处理后的数据导出 (L)。Flink 负责转换 (T) 和复杂事件处理。

您可能会问，为什么会使用 Kafka Connect 目标而不是 Flink 的原生目标（例如 `JdbcSink` 或 `StreamingFileSink`）。

在以下情况使用 **Kafka Connect**：

1. 目标是一个标准的现成系统（Snowflake, Elasticsearch, MongoDB）。
2. 您需要一个仅限配置的解决方案，而无需为连接变更重新部署 JAR 包。
3. 连接器实现已成熟，并能处理复杂的重试/退避逻辑（例如 S3 分区）。

在以下情况使用 **Flink 目标**：

1. 逻辑需要跨越处理和目标（两阶段提交）的事务保证。
2. 目标需要现有连接器中不具备的自定义逻辑。
3. 您希望消除独立 Connect 集群的运维开销。

### 运维弹性

部署 Kafka Connect 需要细致配置内部主题。这些主题存储集群的状态，如果它们被删除或损坏，集群将丢失其配置和偏移量。

- `config.storage.topic`：存储连接器配置。必须具有 `cleanup.policy=compact` 和至少为 3 的 `replication.factor`。
- `offset.storage.topic`：存储源偏移量。必须是压缩的且高副本的。
- `status.storage.topic`：存储连接器和任务的当前状态。

定义连接器时，务必定义一个唯一的 `name`。如果您删除了一个连接器并创建了一个同名的新连接器，它可能会从 `offset.storage.topic` 继承旧的偏移量。要完全重置源连接器，通常必须手动从偏移量主题中删除与该连接器名称关联的特定键。

## 参考资料

- [Apache Kafka Connect Documentation](https://kafka.apache.org/documentation/#connect) — Apache Software Foundation (2023)
  Kafka Connect 的官方综合指南，涵盖其架构、配置、单消息转换 (SMT) 和操作方面。
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043072/) — Gwen Shapira, Todd Palino, Rajini Sivaram, Krit Petty (2021)
  Publisher: O'Reilly Media
  一本关于 Apache Kafka 的综合性书籍，其中包含专门章节，详细解释了 Kafka Connect 的分布式模型、连接器和操作事项。
- [Designing Data-Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于构建可靠、可扩展和可维护数据系统的基础书籍，其中讨论了数据一致性、交付语义和幂等性，对于理解接收器连接器保证很重要。

---

[上一节](02-%E6%A8%A1%E5%BC%8F%E6%B3%A8%E5%86%8C%E8%A1%A8%E9%9B%86%E6%88%90.md) · [下一节](04-%E6%95%85%E9%9A%9C%E6%81%A2%E5%A4%8D%E7%AD%96%E7%95%A5.md)
