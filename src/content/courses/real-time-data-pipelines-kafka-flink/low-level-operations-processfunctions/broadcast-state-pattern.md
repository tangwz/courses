---
course: "real-time-data-pipelines-kafka-flink"
chapter: "low-level-operations-processfunctions"
lesson: "broadcast-state-pattern"
sourceId: 8123
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-5-low-level-operations-processfunctions/broadcast-state-pattern"
title: "广播状态模式"
description: "使用广播状态将配置或规则动态分发到所有并行实例。"
order: 3
plots: ["plots/8123-0.json"]
sourceHash: "9a84f784f537d98ac48d1c8fb8f9b55de29058aad73f3340379b1f85fbb1a219"
sourceCorrections: []
---

流处理管道中的静态配置对于需要灵活性的生产环境来说常常不足够。许多实时场景要求业务逻辑适应不断变化的条件，而不会产生停止、重新编译和重新部署作业所带来的停机时间。Apache Flink 中的广播状态模式通过允许低吞吐量 (throughput)的控制流动态更新应用于高吞吐量数据流的处理逻辑来满足这一需求。

此模式根本上改变了状态的分布方式。与键控状态（根据键的哈希值划分数据）不同，广播状态将状态复制到算子的每个并行实例。这种机制允许您将一个无键流（控制流）与一个带键或无键的数据流连接起来，使所有处理任务同时获取相同的规则或配置数据。

### 广播流的架构

其架构构建方式是将数据摄取与逻辑定义分离。通常有两条流：

1. **事件流：** 主要数据量（例如：交易、日志、传感器读数）。
2. **广播流：** 包含更新的较低数据量流（例如：欺诈规则、阈值、功能开关）。

当广播流连接到事件流时，Flink 确保广播流中的每个元素都被传输到下游算子的所有并行实例。事件流会根据其键继续分区，如果未设置键则直接转发。

> 控制流将数据复制到所有下游任务槽，确保分区数据流上逻辑应用的一致性。

### 实现 KeyedBroadcastProcessFunction

为实现此模式，您可以使用 `KeyedBroadcastProcessFunction`。此函数将一个带键流与一个广播流合并，并需要实现两个方法：`processBroadcastElement` 和 `processElement`。

首先，您必须定义一个 `MapStateDescriptor`。与其他状态类型不同，广播状态必须始终是 Map 形式。这种结构允许算子在处理事件时高效查找特定的配置项。

```java
// 定义广播状态的描述符
MapStateDescriptor<String, Rule> ruleStateDescriptor = new MapStateDescriptor<>(
    "RulesBroadcastState",
    BasicTypeInfo.STRING_TYPE_INFO,
    TypeInformation.of(new TypeHint<Rule>() {})
);

// 广播规则流
BroadcastStream<Rule> broadcastRules = ruleStream
    .broadcast(ruleStateDescriptor);

// 连接并处理
dataStream
    .keyBy(data -> data.getUserId())
    .connect(broadcastRules)
    .process(new KeyedBroadcastProcessFunction<String, Transaction, Rule, Alert>() {

        @Override
        public void processBroadcastElement(Rule rule, Context ctx, Collector<Alert> out) throws Exception {
            // 使用新规则更新状态
            ctx.getBroadcastState(ruleStateDescriptor).put(rule.name, rule);
        }

        @Override
        public void processElement(Transaction txn, ReadOnlyContext ctx, Collector<Alert> out) throws Exception {
            // 读取规则并将其应用于交易
            ReadOnlyBroadcastState<String, Rule> state = ctx.getBroadcastState(ruleStateDescriptor);
            for (Map.Entry<String, Rule> entry : state.immutableEntries()) {
                if (entry.getValue().matches(txn)) {
                    out.collect(new Alert(txn, entry.getKey()));
                }
            }
        }
    });
```

这两种方法中的访问级别存在一个值得注意的区别。在 `processBroadcastElement` 中，您可以使用 `ctx.getBroadcastState` 对广播状态进行读写访问。在这里，您可以根据新的控制消息更新内部映射。

在 `processElement` 中，上下文 (context)只提供 `ReadOnlyBroadcastState`。实行此限制是为了保持确定性。由于数据流在不同节点上并行处理，允许数据流修改广播状态会导致不一致，即不同任务持有不同版本的“全局”配置。只有广播流（它被保证到达所有节点）才允许改变此状态。

### 内存管理和扩展性

虽然广播状态功能强大，但它会产生一种特定的内存占用模式，与键控状态不同。对于键控状态，总状态大小大致与数据中唯一键的数量成比例。如果您扩展集群，状态会被重新分配，并且每个节点的内存压力会降低。

对于广播状态，每个任务都维护广播数据的完整副本。如果您的配置数据增长到 1GB，并且并行度为 100，您的集群实际上会在内存中存储 100GB 的冗余数据。因此，此模式适用于低吞吐量 (throughput)、小尺寸数据集（配置、有限规则集、汇率），而不是大型数据集。

以下图表说明了随着并行度增加的内存消耗行为。



![集群内存使用：键控状态 vs. 广播状态](plots/8123-0.json)



> 随着并行度的扩展，广播状态（红色）在集群中线性增加内存使用量，而分区键控状态（蓝色）的总大小保持不变。

### 检查点和重新扩展行为

Flink 在检查点过程中管理广播状态，通过确保状态是算子快照的一部分。然而，因为状态在所有并行实例中都是相同的，Flink 优化了检查点存储。它不一定需要在分布式文件系统中存储 $P$ 份副本，但逻辑恢复必须保证每个任务在恢复时都拥有该状态。

当作业重新扩展时（例如：并行度从 2 增加到 4），Flink 确保新子任务使用当前广播状态进行初始化。这是通过将现有检查点中的状态复制到新任务实例来实现的。这种行为保证新的处理槽立即拥有处理传入数据所需的完整规则集或配置，而无需等待控制流的重新传输。

### 异步处理

需要注意的是，虽然 Flink 保证单个流内的元素按顺序处理，但广播流和数据流之间的相对顺序在不同任务之间并非严格确定，除非通过水印同步，而广播状态默认不进行原生对齐 (alignment)。

在高速环境中，可能发生竞态条件，即交易到达的毫秒与规则更新广播的毫秒相同。一个任务可能使用旧规则处理交易，而另一个任务可能使用新规则处理类似交易，根据网络到达时间。如果需要控制更新和数据事件之间的严格排序，您必须依赖事件时间，并可能在 `KeyedProcessFunction` 中缓冲数据，直到水印通过配置更改的时间戳，但这会带来很高的延迟和处理难度。

对于大多数应用场景，例如动态过滤、A/B 测试路由或欺诈检测，规则应用的最终一致性是可以接受的，前提是规则传播以亚秒级延迟发生，这是标准 Flink 广播能够实现的。

## 参考资料

- [State & Fault Tolerance](https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/dev/datastream/fault-tolerance/state/) — The Apache Flink Community (2024)
  提供了 Flink 状态机制的基础信息，包括键控状态、算子状态、检查点和保证。
- [Apache Flink: Stream Processing for Real-time Applications](https://www.oreilly.com/library/view/apache-flink/9781491974292/) — Fabian Hueske and Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  由 Flink 核心开发者撰写的综合指南，涵盖了 Flink 的架构、API、状态管理模式（包括广播状态）和部署。
