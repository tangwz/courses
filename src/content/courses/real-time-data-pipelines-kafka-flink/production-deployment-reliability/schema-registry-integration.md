---
course: "real-time-data-pipelines-kafka-flink"
chapter: "production-deployment-reliability"
lesson: "schema-registry-integration"
sourceId: 8127
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/schema-registry-integration"
title: "模式注册表集成"
description: "使用模式注册表强制执行模式兼容性和管理。"
order: 2
plots: ["plots/8127-0.json"]
sourceHash: "211e70395cc300f5b6eef201f96eb0d8d7596d881a0a7802086b04f70d7b7f4d"
sourceCorrections: []
---

数据生产者与消费者解耦是可伸缩分布式系统的基本要求。在本地环境中，生产者和消费者之间共享模式定义文件非常简单。然而，在处理数TB数据的生产流处理架构中，手动模式协调会导致不稳定的数据管道。如果生产者修改数据结构，例如重命名字段或将数据类型从`int`更改为`long`，而所有下游消费者没有同步更新，则反序列化过程将失败。这会产生“毒丸”记录，可能导致整个 Flink 作业停止，或用无法处理的数据填满死信队列。

模式注册表通过充当集中式管理层来解决这个版本控制问题。它存储了 Kafka 集群中使用的所有模式的版本历史。生产者不将完整的模式定义嵌入 (embedding)到每条消息中，而是与注册表通信以验证模式并获取一个唯一标识符。然后将此标识符嵌入到消息头中，与传输冗长的 JSON 或 XML 结构相比，这大大减少了网络开销。

### 线协议与序列化流程

当 Flink 和 Kafka 使用模式注册表时，序列化过程会偏离标准二进制编码。序列化器不仅仅将对象转换为字节；它会在有效载荷前添加一个前导码。

例如，Confluent 模式注册表的线格式由五个初始字节和实际数据组成：

1. **魔术字节（1 字节）：** 一个常数（通常为 `0`），指示协议格式。
2. **模式 ID（4 字节）：** 一个 32 位整数，表示注册表中模式的唯一标识符。
3. **数据：** 序列化后的 Avro、Protobuf 或 JSON 有效载荷。

从数学上看，如果 $S_{schema}$ 是模式定义的大小， $S_{data}$ 是二进制数据的大小，那么在每条消息中嵌入 (embedding)模式会导致总传输大小为 $T = S_{schema} + S_{data}$。通过使用注册表，传输大小变为 $T = 5 \text{ 字节} + S_{data}$。由于 $S_{schema}$ 对于复杂记录通常超过 1KB，因此在高吞吐量 (throughput)下，带宽节省非常可观。

以下图表说明了 Flink 生产者、模式注册表和 Flink 消费者之间的交互。

> 交互流程，说明模式 ID 如何解析模式定义，并通过本地缓存减少对注册表的网络请求。

### 兼容性策略

定义模式如何演进是部署注册表时最主要的配置选择。这些规则，被称为兼容模式，决定了生产者是否可以基于与之前版本的差异来注册新版本的模式。

在数据通常从主题开头重放（使用 `auto.offset.reset = earliest`）的流处理场景中，强兼容性是不可妥协的。

#### 向后兼容性

这是默认且最常用的模式。如果*新*模式可以读取用*旧*模式写入的数据，则该模式是向后兼容的。

- **含义：** 必须在更新生产者之前更新消费者。
- **允许的更改：** 删除字段，添加可选字段（带有默认值）。
- **不允许的更改：** 重命名字段，添加必填字段。

#### 向前兼容性

如果*旧*模式可以读取用*新*模式写入的数据，则该模式是向前兼容的。

- **含义：** 必须在更新消费者之前更新生产者。
- **允许的更改：** 添加字段，删除可选字段。
- **不允许的更改：** 重命名字段，删除必填字段。

#### 完全兼容性

该模式既向后兼容又向前兼容。这提供了最大的灵活性，允许生产者和消费者以任何顺序升级，但对模式修改施加了最严格的限制（例如，字段必须始终具有默认值）。

策略选择会影响您的操作部署顺序。如果您选择**向后**兼容性，则必须升级您的 Flink 消费者，重新启动作业，并确保它们稳定后再部署新的 Flink 生产者应用。

### Flink 实现模式

为了在 Flink 中集成模式注册表，您需要避免使用基本的 `SimpleStringSchema` 或 `JSONKeyValueDeserializationSchema`。相反，您应该使用 Flink Avro 或 Confluent 库提供的特定格式的 serde（序列化器/反序列化器）类。

对于 Avro，`ConfluentRegistryAvroDeserializationSchema` 是标准类。它管理与注册表的 HTTP 交互以及模式 ID 的本地缓存。

配置反序列化器时，您必须在 **SpecificRecords** 和 **GenericRecords** 之间做出选择：

1. **SpecificRecords：** 基于 Avro 模式生成的 Java 类。这提供了编译时类型安全。在 Flink 中，对于复杂的业务逻辑，通常更倾向于使用此方法，因为编译器会捕获字段访问错误。
2. **GenericRecords：** 一种类似映射的结构，其中字段通过名称字符串访问。这对于构建平台级工具（如通用数据摄取管道）很有用，这类工具在编译时不需要知道特定的模式结构。

下面是一个为 `Transaction` SpecificRecord 配置带有模式注册表集成的 Kafka 源的例子：

```java
Properties props = new Properties();
props.setProperty("bootstrap.servers", "broker1:9092");
props.setProperty("group.id", "flink-fraud-detector");

String schemaRegistryUrl = "http://schema-registry:8081";

// Configure Deserializer for SpecificRecord
DeserializationSchema<Transaction> deserializer = 
    ConfluentRegistryAvroDeserializationSchema.forSpecific(
        Transaction.class,
        schemaRegistryUrl
    );

FlinkKafkaConsumer<Transaction> consumer = new FlinkKafkaConsumer<>(
    "transactions-topic",
    deserializer,
    props
);

// Ensure offsets are committed to support exactly-once consistency
consumer.setCommitOffsetsOnCheckpoints(true);
```

### 主题名称策略

模式注册表使用一个称为“Subject”的机制来限定模式范围。默认情况下，Confluent 序列化器使用 **TopicNameStrategy**，其中主题名称源自 Kafka 主题名称（例如，`transactions-topic-value`）。

然而，在高级 Flink 架构中，您可能会遇到“多类型主题”问题，即单个 Kafka 主题包含不同类型的事件。`TopicNameStrategy` 在此会失效，因为它期望每个主题只有一个模式。为了解决此问题，您可以配置 **RecordNameStrategy**，它使用 Avro 记录的完全限定名称（例如，`com.company.events.Transaction`）作为主题。

此配置通过序列化器属性传递：

```java
Map<String, String> serdeConfig = new HashMap<>();
serdeConfig.put(AbstractKafkaSchemaSerDeConfig.VALUE_SUBJECT_NAME_STRATEGY, 
    RecordNameStrategy.class.getName());
```

以下图表突出了在模式迁移过程中，不同兼容模式带来的操作限制。



![兼容模式限制与升级路径](plots/8127-0.json)



> 比较不同兼容模式下的操作灵活性。Y 轴表示标准化的兼容性得分，其中 0 表示破坏性更改。

### 处理序列化失败

在分布式环境中，您必须假定传入数据偶尔会因未能通过注册表验证而失败。如果生产者推送的数据带有的模式 ID 消费者注册表客户端无法解析（例如，消费者和注册表之间存在网络分区），或者如果有效载荷损坏，都可能发生这种情况。

Flink 的标准行为是抛出异常并重启任务，从而触发失败循环。为防止这种情况，请将您的反序列化逻辑包装在一个安全的处理层中，或使用“死信队列”（DLQ）模式。尽管 `FlinkKafkaConsumer` 不原生支持用于反序列化错误的 DLQ（它发生在数据进入 Flink 算子链之前），但您可以配置反序列化器来返回一个包装对象（例如 `Either<Error, Data>`）。随后的 `ProcessFunction` 可以将有效数据路由到主数据流，并将错误路由到旁路输出。

这种防御性编程可确保单个模式不匹配不会导致整个处理管道中断，从而为您的生产 AI 和分析工作负载保持高可用性。

## 参考资料

- [Confluent Platform Documentation: Schema Registry Overview](https://docs.confluent.io/platform/current/schema-registry/index.html) — Confluent, Inc. (2024)
  描述Confluent Schema Registry的核心功能、数据传输协议、兼容性规则和主题命名策略。
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936160/) — Neha Narkhede, Gwen Shapira, Todd Palino (2017)
  Publisher: O'Reilly Media
  关于Apache Kafka的基础资源，涵盖其架构、使用方法以及流式管道中数据序列化和模式演进的实践。
- [Apache Flink Documentation: Kafka Connector](https://flink.apache.org/docs/dev/connectors/kafka/) — The Apache Flink Community (2024)
  Publisher: The Apache Software Foundation
  提供将Flink应用程序连接到Kafka的指南，包括配置各种数据格式和模式注册表的序列化器和反序列化器。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  第4章“编码与演进”详细介绍了各种数据编码格式以及模式演进和兼容性的原则，为Schema Registry解决的问题提供了广阔的视角。
