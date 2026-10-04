---
course: "real-time-data-pipelines-kafka-flink"
chapter: "real-time-ai-feature-engineering"
lesson: "feature-store-integration"
sourceId: 8139
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-8-real-time-ai-feature-engineering/feature-store-integration"
title: "特征存储的整合"
description: "将实时算出的特征数据传输到Redis或Cassandra等快速响应的存储中。"
order: 4
plots: ["plots/8139-0.json"]
sourceHash: "3062219dcb18b97cb740425b57d39c28b69fdece8b5de7e48ffc1b2b93ff9fde"
sourceCorrections: []
---

机器学习 (machine learning)模型在部署到生产环境时常会因为训练-服务偏差而面临性能下降。这种现象发生在推理 (inference)时所用的数据分布或特征计算逻辑与训练模型所用的历史数据不同时。特征存储在流处理架构中作为核心界面，通过保证离线训练环境和在线推理环境之间的一致性来解决这种差异。

### 双数据库架构

为了满足快速响应的查询和处理大容量历史分析这些相互竞争的需求，特征存储通常采用双数据库架构。Flink 作为计算引擎，同时填充这两个存储层。

1. **在线存储（热存储）：** 如 Redis、Cassandra 或 DynamoDB 等系统。这些数据库支持高吞吐量 (throughput)、快速响应的点查询（通过键获取值）。推理 (inference)服务查询此存储，以便在毫秒级别获取给定实体（例如用户或设备）的最新特征向量 (vector)。
2. **离线存储（冷存储）：** 对象存储（S3、GCS）或数据仓库（BigQuery、Snowflake）。这一层保存特征值的历史演变。它不可变并按时间分区，使得数据科学家能够生成时间点精确的训练数据集。

在这种架构中，Flink 流水线作为同步机制。当事件流经流时，Flink 计算聚合特征（例如 `clicks_last_5_minutes`），并执行双写操作。

> 数据流显示 Flink 将算出的特征同步到在线存储供即时推理，并同步到离线存储供历史训练数据使用。

### 写入在线存储

写入在线存储时的主要工程难题是保持高吞吐量 (throughput)而不阻塞 Flink 算子。对像 Redis 这样的数据库的每个事件进行同步调用，会将流水线吞吐量限制到网络往返的响应时间。

为了缓解此问题，必须使用 Flink 的异步 I/O API。这使得流处理算子能够处理对外部存储的多个并发请求。结果的顺序由 Flink 保持，确保对相同键的更新以正确的顺序应用。

设计在线存储的写入器时，通常使用“更新或插入”（Upsert）语义。由于在线存储代表*当前*状态，旧值会被覆盖。

$V_{\text{新}} = f(S_{\text{状态}}, e_{\text{传入}})$

其中 $V_{\text{新}}$ 是新特征值，$S_{\text{状态}}$ 是 Flink 内部状态（例如窗口累加器），而 $e_{\text{传入}}$ 是触发事件。

以下代码说明了如何实现一个异步函数来更新 Redis 特征存储。这种方式采用 `AsyncFunction` 接口，将数据库响应时间与流处理吞吐量解耦。

```java
public class RedisFeatureUpdater extends RichAsyncFunction<FeaturePayload, String> {
    private transient RedisClient redisClient;
    private transient StatefulRedisConnection<String, String> connection;

    @Override
    public void open(Configuration parameters) {
        redisClient = RedisClient.create("redis://localhost:6379");
        connection = redisClient.connect();
    }

    @Override
    public void asyncInvoke(FeaturePayload input, ResultFuture<String> resultFuture) {
        // 异步更新特征向量
        RedisAsyncCommands<String, String> async = connection.async();

        // 实体 ID，值：序列化后的特征向量
        CompletionStage<String> future = async.set(
            "feature:" + input.getEntityId(), 
            input.toJson()
        );

        future.whenComplete((res, error) -> {
            if (error != null) {
                resultFuture.completeExceptionally(error);
            } else {
                resultFuture.complete(Collections.singleton(res));
            }
        });
    }
}
```

### 时间点精确性

对离线存储的一项主要要求是，能够在任何特定历史时间戳重建特征状态。这被称为时间旅行。如果一个模型根据过去一小时的交易数量预测欺诈，训练数据必须在欺诈标签生成的那一刻*准确*反映该计数，而不是一天结束时的计数。

当 Flink 写入离线存储时，它必须追加记录而不是更新它们。离线存储中的每条记录应包含：

1. **实体 ID：** 用户或对象标识符。
2. **特征值：** 算出的值（例如 $0.75$）。
3. **事件时间戳：** 事件实际发生的时间（来自源）。
4. **处理时间戳：** Flink 处理记录的实际时间。

离线存储本质上成为一个只追加的特征变更记录。当数据科学家生成训练数据集时，他们执行“as-of join”逻辑：

$\mathcal{F}(e, t_{\text{标签}}) = \text{argmax}_{t \leq t_{\text{标签}}} \text{FeatureLog}(e, t)$

此查询选择实体 $e$ 的最新特征值，该值必须严格早于或等于观察到目标变量的时间 $t_{\text{标签}}$。

### 响应时间和新鲜度权衡

整合特征存储引入了数据新鲜度与系统繁杂程度之间的权衡。在基于批处理的特征工程中，特征定期更新（例如每日），导致一种“锯齿形”新鲜度模式，在这种模式下，数据逐渐变得陈旧，直到下一次批处理运行。流处理整合创造了一条接近平坦的新鲜度线，使事件时间与特征可用性之间的差异最小化。

然而，写入远程存储会引入网络响应时间。优化 Flink 写入器常涉及缓冲写入或使用流水线技术。



![数据新鲜度：批处理与流式特征更新对比](plots/8139-0.json)



> 特征新鲜度对比。批处理更新随时间推移而下降，造成现实与模型视图之间的差异。流处理更新保持接近零的响应时间。

### 处理延迟数据和一致性

在分布式系统中，事件常会乱序到达。Flink 的水印机制在内部处理聚合的精确性，但更新外部特征存储需要特定的策略，以防止“僵尸”数据，即旧事件覆盖在线存储中的新值。

处理这种同步有两种核心策略：

1. **版本检查：** 在线存储的 schema 包含 `last_updated_timestamp` 列。写入器函数使用条件写入（比较并交换）。仅当传入记录的时间戳大于已存储的时间戳时才应用更新。
2. **幂等聚合：** 写入器不是覆盖值，而是应用可交换操作（例如 Redis 中的 `INCRBY`），或存储在查询时求和的原始聚合数据。这能够抵御乱序，但要求读取路径（推理 (inference)服务）执行最终计算。

对于离线存储，延迟数据问题较小，因为它是一个只追加的记录。那里的难题是纯粹的分析问题：确保“as-of join”查询准确考虑数据*何时可用于*推理，以避免训练期间的数据泄露。

## 参考资料

- [Michelangelo: Uber's Machine Learning Platform](https://arxiv.org/pdf/1710.05370) — Siby, Praveen and Goel, Anirudh and Jain, Manu and Jain, Amit and Singh, Amit and Sharma, Hien and Kumar, Karan and Panwar, Vaibhav and Srivatsa, Sathish and Shvachko, Alex and Grewal, Navjot and Arora, Sahitya and Kumar, Manish and Singh, Kamesh and Singh, Gaurav and Pudi, Venkata and Nallamothu, Venkata and Jain, Anupam and Gupta, Sanjai (2017)
  Journal: arXiv preprint arXiv:1710.05370
  描述了大规模机器学习平台的架构，包括特征存储的必要性及其设计，以解决训练-服务偏差并确保特征一致性。
- [Asynchronous I/O for External Data Access](https://flink.apache.org/docs/concepts/datastream/operators/async-io/) — Apache Flink Documentation (2024)
  官方文档解释了 Flink 的异步 I/O API，这对于与外部在线特征存储进行高吞吐量交互至关重要。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Huyen, Chip (2022)
  Publisher: O'Reilly Media
  全面解释了特征存储，涵盖了其架构、在防止训练-服务偏差中的作用以及对机器学习系统点历史一致性等要求。
