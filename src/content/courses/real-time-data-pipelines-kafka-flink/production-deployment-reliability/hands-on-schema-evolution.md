---
course: "real-time-data-pipelines-kafka-flink"
chapter: "production-deployment-reliability"
lesson: "hands-on-schema-evolution"
sourceId: 8130
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-6-production-deployment-reliability/hands-on-schema-evolution"
title: "实战：运行中的模式演进"
description: "更新生产者模式并验证消费者兼容性，无需重启数据管道。"
order: 5
plots: ["plots/8130-0.json"]
sourceHash: "02dd80dbfe7e62bc68ca32bc319a0d2d9ac69fe0d2d3d0a25f85ed1d2fb2d521"
sourceCorrections: []
---

当数据结构保持不变时，维护流式数据管道很简单。当业务需求要求更改事件结构，同时系统每秒处理数千个事件时，操作难度会急剧增加。你不能简单地停止管道，清空主题，然后使用新模式重启。你必须实施一个运行中的模式演进方案。

本节将使用Apache Avro和Confluent Schema Registry进行一次受控的模式升级。我们将模拟一个常见的生产情况：在不中断现有消费者或不强制停止Flink集群的情况下，为分析事件添加一个新的维度。

### 兼容性约定

在编写代码之前，我们必须明确生产者和消费者之间的交互规则。在解耦架构中，这些组件独立演进。Schema Registry强制执行兼容模式，以防止消费者在读取无法识别的格式时崩溃，也就是避免“毒丸”情况。

对于高吞吐系统，**向后兼容性**是优先升级消费者的标准做法。

- **向后兼容性：** 升级到模式V2的消费者可以读取由使用模式V1的生产者写入的数据。
- **工作流程：** 首先升级Flink消费者应用。它会使用缺失字段的默认值处理旧数据格式。一旦消费者池稳定，你就可以升级生产者，使其开始发送新格式的数据。

下图展示了向后兼容性部署期间的数据流。

> 已升级的消费者（V2）与Schema Registry交互，将V1记录投影到V2结构中，确保应用代码不会遇到 `ClassCastException`。

### 步骤1：定义基线模式（V1）

我们从一个简单的`Transaction`事件开始。在一个工作流程中，这个Avro定义会放在`src/main/resources/avro`目录下的一个`.avsc`文件中。

```json
{
  "type": "record",
  "name": "Transaction",
  "namespace": "com.pipeline.events",
  "fields": [
    {"name": "transaction_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "timestamp", "type": "long"}
  ]
}
```

当这个模式注册后，Schema Registry会给它分配一个唯一的ID（例如，ID 1）。Kafka生产者序列化数据并将此ID添加到有效载荷的前面。

### 步骤2：演进模式（V2）

市场部门需要追踪与交易关联的`promotion_code`。为了安全地添加这个字段，我们必须考虑到主题中缺少该字段的现有数据。

如果我们将该字段添加为强制性（非空），V2消费者在尝试读取V1数据时将失败，因为V1数据不包含`promotion_code`。为了保持向后兼容性，我们必须指定一个**默认值**。

下面是演进后的V2模式：

```json
{
  "type": "record",
  "name": "Transaction",
  "namespace": "com.pipeline.events",
  "fields": [
    {"name": "transaction_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "timestamp", "type": "long"},
    {
      "name": "promotion_code", 
      "type": ["null", "string"], 
      "default": null
    }
  ]
}
```

核心的增加是 `"default": null`。这指示Avro反序列化器：“如果你遇到缺少`promotion_code`的记录（即V1记录），用`null`填充该空缺。”

### 步骤3：配置Flink消费者

在Flink中，仅仅从Avro定义重新生成Java类是不够的。你必须配置反序列化器，使其精确地遵循V2模式生成的特定类结构。

使用`FlinkKafkaConsumer`时，需要进行特定配置来启用模式演进支持。

```java
Properties props = new Properties();
props.setProperty("bootstrap.servers", "broker1:9092");
props.setProperty("group.id", "transaction-processor");

// 模式演进配置
props.put(KafkaAvroDeserializerConfig.SPECIFIC_AVRO_READER_CONFIG, "true");
props.put(AbstractKafkaSchemaSerDeConfig.SCHEMA_REGISTRY_URL_CONFIG, "http://registry:8081");

KafkaAvroDeserializationSchema<Transaction> deserializer = 
    new KafkaAvroDeserializationSchema<>(
        Transaction.class
    );

FlinkKafkaConsumer<Transaction> consumer = new FlinkKafkaConsumer<>(
    "transactions",
    deserializer,
    props
);
```

将`SPECIFIC_AVRO_READER_CONFIG`设置为`true`是一种机制，它强制反序列化器将写入者模式（消息中的任何版本）投影到读取者模式（编译到Flink jar中的V2类）上。没有这个设置，消费者将作为一个通用读取器，可能无法正确应用默认值，从而导致运行时错误。

### 步骤4：执行与验证

为了验证升级期间管道的稳定性，我们监控反序列化错误率。成功的部署特点是，在V1和V2模式共存的过渡期内，没有出现序列化异常。

过渡期间数据可用性 $A$ 的数学预期（其中 $N$ 是总记录数，$E$ 是反序列化错误数）应为：

$A = \frac{N - E}{N} = 1$

如果 $E > 0$，则兼容性约定已被违反。

我们可以将升级过程的时间线可视化。目标是确认在任何时候，消费者延迟不会因模式不匹配而失控增加。



![滚动模式升级期间的指标表现](plots/8130-0.json)



> 在消费者（第10分钟）和生产者（第20分钟）升级期间，吞吐量 (throughput)保持不变，这表明模式演进成功且具有向后兼容性。

### 处理模式验证失败

如果你尝试注册一个违反兼容模式的模式（例如，生产者删除了一个强制性字段，而消费者仍然期望它存在），Schema Registry将以`409 Conflict`错误拒绝注册请求。

为了防止这种情况在部署期间发生，你应该在构建工件之前，使用Schema Registry Maven插件或REST API在你的CI/CD管道中运行兼容性检查。

```bash
# 部署前使用Schema Registry API检查兼容性的示例
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{"schema": "{\"type\": \"record\", ...}"}' \
  http://registry:8081/compatibility/subjects/transactions-value/versions/latest
```

如果响应为 `{"is_compatible": true}`，则继续。如果为`false`，则管道停止，保护生产环境免受损坏。

通过坚持严密的向后兼容性方案并利用Avro中的`default`字段，你可以解耦生产者和消费者的生命周期。这使得你的Flink应用能够适应不断变化的业务逻辑，而无需停机或重新处理数据。

## 参考资料

- [Schema Registry and Schema Evolution](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html) — Confluent (2024)
  Publisher: Confluent
  官方文档，说明了模式兼容性规则、Avro 序列化以及 Schema Registry 在管理模式变更以实现数据管道稳定方面的作用。
- [Apache Avro 1.11.1 Specification](https://avro.apache.org/docs/1.11.1/spec.html) — Apache Avro Project (2022)
  Publisher: The Apache Software Foundation
  详细说明了 Avro 数据序列化格式、模式定义以及模式解析规则，包括在反序列化过程中默认值如何处理缺失字段，这对于模式演进至关重要。
- [Kafka: The Definitive Guide, 2nd Edition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPuOY_4J5g2Q53Z9j0k5uC6_9VWqB92ZI0DN0h1Se00KF4dJmitM5Hu87Ieu2r3s8r8c4yBtyYwEpZA64TNmpgmAycGdm4mKme9N7pbqllw1ONagxzdOl2-kjtHJwX385QnYtp2bwDHYiLNHvW2ISkKapmM8eYmPkiMFrRteiilo-) — Neha Narkhede, Gwen Shapira, Todd Palino, Rajini Sivaram (2021)
  Publisher: O'Reilly Media
  提供关于使用 Apache Kafka 构建和管理数据管道的指导，包括数据治理、使用 Avro 进行模式演进以及 Schema Registry 的信息。
