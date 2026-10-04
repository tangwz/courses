---
course: "real-time-data-pipelines-kafka-flink"
chapter: "flink-state-management-checkpointing"
lesson: "hands-on-state-migration-savepoints"
sourceId: 8109
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-3-flink-state-management-checkpointing/hands-on-state-migration-savepoints"
title: "动手实践：使用 Savepoint 进行状态迁移"
description: "使用 Flink Savepoint 执行应用程序升级和扩缩容操作。"
order: 5
plots: []
sourceHash: "f45c0adc0951ab6ec5d87f9c1c728d0127f69a46ad966c05e8c94b0d17070763"
sourceCorrections: []
---

生产环境需要持续迭代。业务逻辑会演变，错误需要修补，流量高峰要求扩缩容。在无状态架构中，更新很简单：替换二进制文件并重启服务即可。在有状态流处理中，此操作会销毁内存上下文 (context)、滚动聚合、会话窗口以及数周或数月累积的机器学习 (machine learning)模型权重 (weight)。Flink 通过 **Savepoint** 将处理图与其状态分开，这使你能够暂停应用程序，将其内存保存到分布式文件系统，并在以后恢复，即使代码已修改或并行度不同。

### Savepoint 与 Checkpoint 的运行机制

尽管 Checkpoint 和 Savepoint 都使用相同的底层快照机制（异步屏障快照），但它们的生命周期和目的不同。Checkpoint 由 Flink 管理，用于故障时的自动恢复。它们是临时的，归运行时所有。Savepoint 是手动触发的，归用户所有，并会持续存在直到被明确删除。它们充当流处理执行状态在特定时间点的一个可移植镜像。

为实现状态迁移，Flink 将 Savepoint 中的序列化字节映射到新应用图中的算子。此映射依赖于算子标识。默认情况下，Flink 根据拓扑结构生成算子 ID。如果你向流水线中插入一个通用 `filter()`，所有后续算子的 ID 都会改变。当新作业尝试从 Savepoint 恢复时，它将无法将状态与新拓扑匹配，从而导致数据丢失或启动失败。

### 明确的算子标识

应对迁移过程中拓扑不匹配的主要方法是显式分配唯一 ID (UID)。你必须为 DataStream API 链中的每个有状态算子分配一个稳定的标识符。

考虑一个简化的欺诈检测流水线。以下代码展示了如何将 UID 附加到负责维护用户配置文件的 `KeyedProcessFunction`。

```java
DataStream<Transaction> transactions = env.addSource(kafkaSource)
    .name("Kafka 源")
    .uid("source-id"); // 显式 UID

DataStream<Alert> alerts = transactions
    .keyBy(Transaction::getUserId)
    .process(new FraudDetector())
    .name("欺诈检测器")
    .uid("fraud-detector-id"); // 对于状态迁移非常必要

alerts.addSink(kafkaSink)
    .name("Kafka 汇")
    .uid("sink-id");
```

如果你省略 `.uid("fraud-detector-id")`，Flink 会根据算子的位置生成一个哈希值。在未来的版本中，如果在处理函数之前添加一个 `map()` 操作，将改变该哈希值，从而使之前的状态无法访问。

### 执行：迁移工作流程

我们将介绍一个标准迁移场景：同时进行作业扩容和处理逻辑更新。

#### 1. 触发 Savepoint

假设作业正在运行，其作业 ID 为 `a1b2c3d4`。你发出命令，将 Savepoint 触发到持久化存储位置（如 S3 或 HDFS）。

```bash
$ flink savepoint a1b2c3d4 s3://flink-savepoints/fraud-detection/
正在为作业 a1b2c3d4 触发 Savepoint。
等待响应...
Savepoint 已完成。路径：s3://flink-savepoints/fraud-detection/savepoint-a1b2c3-8d9e0f
```

返回的路径包含元数据文件以及对检查点数据文件（如果使用 RocksDB，则是 SST 表）的引用。

#### 2. 停止作业

一旦 Savepoint 确认完成，你就可以安全地取消正在运行的作业。在现代 Flink 版本中，你可以使用 `stop` 命令的 `-s` 标志合并这些步骤，这能确保在 Savepoint 完成和作业终止之间没有数据被处理。

```bash
$ flink stop --savepointPath s3://flink-savepoints/fraud-detection/ a1b2c3d4
```

#### 3. 更新应用程序

你现在修改 `FraudDetector` 类。也许你将逻辑更改为标记 (token)超过 500 美元而不是 100 美元的交易。你还决定将并行度从 4 增加到 8 以处理增加的负载。

由于我们使用了 `.uid("fraud-detector-id")`，Flink 可以在 Savepoint 中找到 `FraudDetector` 算子的状态，即使我们重新排序图中的算子，只要状态数据类型（模式）保持兼容。

#### 4. 以新状态恢复

提交新的 JAR 文件，将其指向 Savepoint 路径。

```bash
$ flink run -s s3://flink-savepoints/fraud-detection/savepoint-a1b2c3-8d9e0f \
  -p 8 \
  -c com.example.FraudJob \
  new-fraud-job.jar
```

`-p 8` 标志指示 Flink 进行扩缩容。状态后端（RocksDB 或 HashMap）会将键重新分布到新的工作节点上。

### 扩缩容和键组

扩缩容不是简单的文件复制。Flink 将状态分区为 **键组**。键组的数量由 `maxParallelism` 参数 (parameter)决定（默认为 128，但可配置）。每个传入的键使用以下公式分配给一个键组：

$\text{键组} = \text{murmurHash}(\text{键}) \pmod{\text{maxParallelism}}$

键组是状态分配的原子单元。当并行度为 $P$ 时，每个并行实例都被分配一个键组范围。

$\text{每个实例的键组数} \approx \frac{\text{maxParallelism}}{P}$

当我们从 $P=4$ 扩容到 $P=8$ 时，Flink 会将分配给原始 4 个实例的键组进行拆分，并重新分配给 8 个新实例。此逻辑编码在 Savepoint 元数据中，允许新实例仅拉取与其分配的键组相关的特定 SST 文件（对于 RocksDB）。

> 状态在扩缩容操作期间的重新分布。状态通过键组进行追踪，从而能够灵活地重新分配给新任务。

### 处理模式演变

迁移过程中一个常见挑战是改变状态中存储的数据结构，例如，向 `FeatureState` POJO 添加一个 `riskScore` 字段。

如果你使用 Flink 的 POJO 序列化器或 Avro，系统在一定程度上支持模式演变。

1. **POJO 序列化器：** 如果严格遵循 POJO 规则（例如，默认构造函数、getter/setter），Flink 可以处理添加或删除的字段。恢复时，Flink 会删除已删除的字段，并将新字段初始化为其默认 Java 值（null、0、false）。
2. **Avro：** 如果你使用 Avro 生成的类作为状态，则适用标准的 Avro 模式演变规则。你可以添加带有默认值的字段。

但是，如果你使用自定义二进制序列化器或 Kryo 回退机制，模式更改将是二进制不兼容的。恢复将因反序列化异常而失败。在此类高级情况中，你必须使用 **状态处理器 API**，这是一个基于数据集的 API，它允许你将 Savepoint 读取为批处理集合，使用 Map/Reduce 转换处理二进制状态，并为新作业写入新的有效 Savepoint。

### 运维最佳实践

1. **始终分配 UID：** 将此作为强制性代码审查清单项。没有 UID，你就无法改变拓扑。
2. **监控恢复延迟：** 从 S3 恢复大量状态（数 TB）可能因网络带宽而耗时。确保 `taskmanager.network.memory` 缓冲区足以处理状态重新分布期间的数据混洗。
3. **规范最大并行度：** 显式设置 `maxParallelism`（例如 4096），而不是让 Flink 自动推导。如果此值改变，你将无法恢复状态，因为键组映射计算 $\text{哈希} \pmod{\text{maxParallelism}}$ 会改变。
4. **本地恢复：** 对于 RocksDB，请确保在 `flink-conf.yaml` 中启用本地恢复。这允许 TaskManager 从状态的本地副本（如果磁盘完好）恢复，而不是从远程存储下载完整的 Savepoint，从而显著减少重启时间。

通过掌握 Savepoint，你可以将流处理流水线从脆弱、短暂的进程转变为一个有弹性、可演进的数据系统。你可以修补逻辑、升级集群，并为 A/B 测试分叉流，而不会损害你的长期聚合的完整性。

## 参考资料

- [Savepoints](https://flink.apache.org/docs/latest/docs/ops/state/savepoints/) — Apache Flink Community (2024)
  介绍了Apache Flink中Savepoint的用途、机制以及在应用程序升级和迁移中的使用。
- [The State Processor API](https://flink.apache.org/2019/09/13/state-processor-api.html) — Apache Flink Community (2019)
  Publisher: Apache Software Foundation
  说明如何使用批处理从Savepoint或Checkpoint读取、修改和写入Flink状态，这对于状态模式演变特别有用。
- [Streaming Systems: The What, Where, When, and How of Large-Scale Data Processing](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) — Tyler Akidau, Slava Chernyak, Reuven Lax (2018)
  Publisher: O'Reilly Media; Pages: 349
  提供了大规模流处理系统中状态化流处理、一致性模型和容错的原理。
- [Apache Flink: Stream Processing at Scale](https://dl.acm.org/doi/10.14778/2824032.2824047) — Paris Carbone, Asterios Katsifodimos, Stephan Ewen, Volkan Özcelik, Kostas Tzoumas, and Daniel Warneke (2015)
  Journal: Proceedings of the VLDB Endowment; Publisher: The VLDB Endowment; Volume: 8; Pages: 1777-1788; DOI: [10.14778/2824032.2824047](https://doi.org/10.14778/2824032.2824047)
  介绍了Apache Flink的架构和核心组件，包括其状态管理和容错机制。
