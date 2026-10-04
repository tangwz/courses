# Avro 和 Protobuf 序列化

来源：[原文](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/serialization-avro-protobuf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将数据管道从受控的开发环境扩展到高吞吐量 (throughput)的生产集群时，会显现出在低数据量时不易察觉的瓶颈。分布式流处理中最常见的性能瓶颈之一是序列化和反序列化的开销。当每秒在 Kafka 代理和 Flink 任务槽之间传输数 GB 数据时，数据的格式决定着网络饱和度和 CPU 占用率。尽管像 JSON 这样的文本格式因其可读性和灵活性而普遍存在，但它们不适合实时 AI 和分析系统的高要求。

### 冗余的代价

JSON 是自描述的，意味着每条记录都以字段名称的形式带有自己的 schema。假设一个简单的传感器读数。在 JSON 中，单个事件可能如下所示：

```json
{"sensor_id": "s-101", "timestamp": 1610000000, "temperature": 23.5}
```

这个有效载荷大约占用 65 字节。实际数据值、标识符、长整型时间戳和浮点型温度只占其中很小一部分。字段名（`sensor_id`、`timestamp`）在数百万个事件中的重复传输浪费带宽，并要求 CPU 对每一条消息解析字符串标记 (token)。

相比之下，像 Apache Avro 和 Protocol Buffers (Protobuf) 这样的二进制序列化格式会从有效载荷中移除这种结构元数据。它们不传输字段名，而是依赖于预定义的 schema 将位映射到数据结构。这样生成的有效载荷通常会小 40% 到 60%。

数学上，如果 $S_{msg}$ 是平均消息大小，$R$ 是每秒消息速率，则所需的网络带宽 $BW$ 为：

$BW = S_{msg} \times R$

将 $S_{msg}$ 减半可以有效地使您现有网络基础设施的吞吐量 (throughput)容量加倍，无需增加硬件。

### 流处理中的 Apache Avro

Apache Avro 是在 Hadoop 生态系统内发展起来的一种面向行的远程过程调用和数据序列化框架。由于其紧凑的二进制格式以及对 schema 演进的良好支持，它对数据密集型应用尤其有用。

在 Flink 和 Kafka 的情境中，Avro 通常是默认选择，因为它与 Confluent Schema Registry 集成良好。Avro 依赖于 JSON 定义的 schema 来解释数据。然而，与标准 JSON 处理不同，schema 不随每条消息发送。在基于 Kafka 的架构中，生产者向注册中心注册 schema 并接收一个唯一的 4 字节整数 ID。此 ID 加到二进制消息之前。消费者读取 ID，从注册中心获取 schema（并本地缓存），然后反序列化有效载荷。

一个典型的 Avro schema 定义如下：

```json
{
  "namespace": "com.pipeline.events",
  "type": "record",
  "name": "SensorReading",
  "fields": [
    {"name": "sensor_id", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "temperature", "type": "float"}
  ]
}
```

当 Flink 处理这些记录时，它可以通过两种模式处理它们：

1. **特定记录 (SpecificRecord):** 您在构建过程中根据 schema 生成 Java/Scala POJO 类。这提供编译时类型安全性和更好的性能，因为字段访问是直接的。
2. **泛型记录 (GenericRecord):** Flink 将数据作为泛型映射结构处理。这对必须处理任意数据类型的管道有用，但由于查找操作会带来性能损失。

### Protocol Buffers (Protobuf)

由 Google 开发的 Protocol Buffers 提供了目标相似但方法不同的方式。Protobuf 要求您在 `.proto` 文件中定义消息。与 Avro 基于 JSON 的 schema 不同，Protobuf 使用自定义接口描述语言 (IDL)。

```protobuf
syntax = "proto3";

package com.pipeline.events;

message SensorReading {
  string sensor_id = 1;
  int64 timestamp = 2;
  float temperature = 3;
}
```

分配给字段的整数（例如 `sensor_id = 1`）用作标签。在序列化过程中，Protobuf 写入字段标签和线类型，然后是值。这种结构允许跳过字段（如果它们未知或已弃用），提供向前和向后兼容性。

Protobuf 在序列化和反序列化 (SerDes) 方面通常比 Avro 更快，因为生成的代码针对特定语言进行了高度优化。它在 CPU 周期是限制因素的环境中表现出色。然而，Avro 通常为批量存储（如 S3 parquet）生成略小的文件，因为它在写入数据块时不会为每个值存储字段标签。

### 性能对比

Avro 和 Protobuf 之间的选择通常会考虑到其所处的生态系统，而非纯粹的性能，因为两者都优于 JSON。然而，了解它们的资源配置文件有助于调整 Flink 集群。

> 跨格式的序列化大小和处理时间比较。与文本格式相比，二进制格式显著减少网络负载和 CPU 时间。



[交互图表：序列化格式基准（标准化）](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/serialization-avro-protobuf#plot-se6cz8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "序列化格式基准（标准化）",
    "xaxis": {
      "title": "格式"
    },
    "yaxis": {
      "title": "标准化得分（越低越好）"
    },
    "barmode": "group",
    "width": 600,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 50,
      "t": 50,
      "b": 50
    },
    "legend": {
      "orientation": "h",
      "y": -0.2
    }
  },
  "data": [
    {
      "x": [
        "JSON",
        "Avro",
        "Protobuf"
      ],
      "y": [
        100,
        55,
        52
      ],
      "name": "有效载荷大小",
      "type": "bar",
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "x": [
        "JSON",
        "Avro",
        "Protobuf"
      ],
      "y": [
        100,
        65,
        40
      ],
      "name": "序列化时间（CPU）",
      "type": "bar",
      "marker": {
        "color": "#9775fa"
      }
    }
  ]
}
```

</details>



### 在 Flink 中实现序列化

为了保持精确一次语义和系统稳定，Flink 必须知道在操作符任务之间传递数据时（例如，在 `keyBy` shuffle 期间）以及将状态检查点到 RocksDB 时如何序列化数据。

默认情况下，如果 Flink 无法确定一个复杂对象的类型，它会退回到 Kryo 序列化器。Kryo 是一个通用的 Java 序列化框架。尽管它很灵活，但它显著更慢，并产生比 Avro 或 Protobuf 更大的二进制数据块。在生产环境中，您必须主动避免 Kryo 回退。

#### 配置 Avro 序列化器

要在 Flink 的 DataStream API 中使用 Avro，您使用 `AvroSerializationSchema` 进行 Kafka 写入，并使用 `AvroDeserializationSchema` 进行读取。

定义 Flink source 时，您将反序列化器配置为严格地将传入字节映射到您生成的类。

```java
KafkaSource<SensorReading> source = KafkaSource.<SensorReading>builder()
    .setBootstrapServers("broker:9092")
    .setTopics("sensor-readings")
    .setGroupId("pipeline-group")
    .setValueOnlyDeserializer(
        AvroDeserializationSchema.forSpecific(SensorReading.class)
    )
    .build();
```

使用 `AvroDeserializationSchema.forSpecific` 指示 Flink 跳过泛型解释并直接实例化 POJO。这为 Flink 的类型提取器提供了足够的信息，以便使用其高效的内部序列化器（POJO 序列化器）进行状态管理，而不是回退到 Kryo。

#### 在 Flink 中处理 Protobuf

Flink 在其连接器中提供了 `FlinkProtobufSchema`。与 Avro 相似，利用生成的 Protobuf 类可以确保 Flink 识别数据结构。

如果您正在使用 Protobuf，您必须确保 `protoc` 编译器版本与您的 Flink fat JAR 中使用的运行时库版本匹配。版本不匹配是部署期间 `InvalidProtocolBufferException` 错误的常见原因。

### 生产最佳实践

1. **优先使用特定记录 (Specific Records):** 始终为您的数据类型生成特定类（POJO）。避免在核心转换逻辑中使用泛型记录，因为类型转换和映射查找会降低吞吐量 (throughput)。
2. **禁用 Kryo 回退:** 在您的 Flink 执行环境配置中，如果作业尝试使用 Kryo，您可以强制其失败。这是一个确保所有类型都得到高效处理的妥善方式。

   ```java
   env.getConfig().disableGenericTypes();
   ```
3. **Schema Registry 是必要的:** 永远不要在流消息的有效载荷中嵌入 (embedding) schema。始终使用外部 Schema Registry 将 ID 映射到 schema 定义。这将生产者的演进生命周期与消费者的逻辑分离。

通过强制使用二进制序列化，您可以减少 Kafka 代理上的 I/O 压力，并释放 Flink TaskManager 上的 CPU 周期，使系统能够将资源用于实际的计算和状态管理任务。

## 参考资料

- [Apache Avro™ 1.11.1 Documentation](https://avro.apache.org/docs/current/) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  Apache Avro的官方文档，提供了其数据序列化格式和RPC协议的规范。
- [Protocol Buffers](https://developers.google.com/protocol-buffers) — Google (2024)
  Google Protocol Buffers的官方文档，详细介绍了其语言中立、平台中立、可扩展的结构化数据序列化机制。
- [Confluent Platform Documentation - Schema Registry](https://docs.confluent.io/platform/current/schema-registry/index.html) — Confluent (2024)
  Confluent Schema Registry的详细指南，涵盖了其在Kafka数据管道中管理Avro模式的作用。
- [Apache Flink Documentation - Type Information and Serialization](https://nightlies.apache.org/flink/flink-docs-master/docs/dev/datastream/fault-tolerance/serialization/types_serialization/) — The Apache Software Foundation (2024)
  Flink官方文档，关于数据类型如何处理和序列化，包括特定序列化器的使用和避免Kryo回退。
- [Kafka: The Definitive Guide: Real-time Data and Stream Processing at Scale](https://books.google.com/books/about/Kafka_The_Definitive_Guide.html?id=lB-NDgAAQBAJ) — Gwen Shapira, Neha Narkhede, and Todd Palino (2017)
  Publisher: O'Reilly Media
  涵盖了Kafka的数据序列化策略，包括二进制格式的优势和Schema Registry的作用。

---

[上一节](../05-%E4%BD%BF%E7%94%A8%20ProcessFunction%20%E7%9A%84%E5%BA%95%E5%B1%82%E6%93%8D%E4%BD%9C/05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%99%E8%AF%84%E4%BC%B0.md) · [下一节](02-%E6%A8%A1%E5%BC%8F%E6%B3%A8%E5%86%8C%E8%A1%A8%E9%9B%86%E6%88%90.md)
