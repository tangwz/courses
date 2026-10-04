---
course: "real-time-data-pipelines-kafka-flink"
chapter: "performance-tuning-monitoring"
lesson: "memory-management-slot-allocation"
sourceId: 8132
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-7-performance-tuning-monitoring/memory-management-slot-allocation"
title: "内存管理与槽位分配"
description: "配置 TaskManager 内存段、网络缓冲区和槽位共享组。"
order: 2
plots: ["plots/8132-0.json"]
sourceHash: "75dce5ae449777291cfe99a7497ee53253b250e254f5fe7b573a8105a9da2cac"
sourceCorrections: []
---

Apache Flink 在一个内存层级结构中运行，该结构远超出标准 Java 虚拟机堆的配置。将 Flink 应用程序部署到生产环境时，您并非简单地为 Java 进程分配堆大小；您是在配置复杂的、独立的内存段，这些内存段用于处理网络传输、框架开销、用户代码执行以及堆外状态存储。此处的配置不当是高吞吐量 (throughput)数据管道中容器重启和 `OutOfMemoryError` 故障的主要原因。

弄清楚 TaskManager 的具体作用是优化的第一步。TaskManager 是 Flink 中的工作进程。尽管 JVM 堆存储着用户自定义函数创建的 Java 对象，但 Flink 会为内部操作主动管理独立的内存池。这在使用 RocksDB 状态后端时尤为重要，因为它依赖于 JVM 堆之外的原生内存。

### Flink 内存模型

分配给容器或进程的总内存定义为 *总进程内存*。Flink 将其划分为 *Flink 总内存*（Flink 实际控制的内存）和 *JVM 开销*（为 JVM 自身预留的内存，包括 Metaspace 和线程栈）。

在 *Flink 总内存* 内部，其划分决定了性能表现：

1. **JVM 堆：** 由框架堆（内部结构）和任务堆（用户代码对象）组成。
2. **堆外内存：** 包括托管内存和直接内存。
3. **网络缓冲区：** 专用于算子间数据交换的内存段。

下图详细说明了这种层级结构。请注意托管内存是如何位于 JVM 堆之外的。

> TaskManager 内存模型的分解图，显示了堆内存和堆外内存段之间的分离。

**托管内存** 是有状态流处理中最重要的部分。使用 RocksDB 时，Flink 会将此内存段分配给嵌入 (embedding)式数据库，用于块缓存和写入缓冲区。如果此内存段过小，RocksDB 将难以缓存数据，导致频繁的磁盘读取，从而增加延迟。如果它相对于 JVM 堆过大，您的用户代码可能在垃圾回收峰值期间崩溃。

默认情况下，Flink 将 `taskmanager.memory.managed.fraction` 设置为 0.4。这表示 40% 的 Flink 总内存会保留给 RocksDB 使用。在状态庞大但转换逻辑简单的场景中，将其增加到 0.6 或 0.7 通常能提高稳定性。相反，如果您的逻辑涉及复杂的窗口聚合，且这些聚合在堆中持有对象（例如 `ListState`），您必须降低此比例以优先保障任务堆的内存。

### 网络缓冲区与吞吐量 (throughput)

任务之间移动的数据，从 Map 算子分区到 Reduce 算子，必须通过网络缓冲区。Flink 使用一种基于信用点的流控制机制，接收方授予发送方信用点来传输数据。这可以防止发送方使接收方过载。

这些缓冲区位于堆外直接内存中。配置参数 (parameter) `taskmanager.memory.network.fraction` 控制其大小。默认值为 0.1（10%）。

在高吞吐量环境中，网络内存不足会导致“缓冲区超时”情况，数据管道会停滞，等待内存段释放。这会表现为反压。如果您观察到 CPU 使用率高但吞吐量低，请检查您的网络缓冲区是否已满。您可以使用并发连接数粗略估算网络缓冲区所需的内存：

$网络内存 \approx 槽位数量 \times 对等连接数 \times 缓冲区大小$

如果您的拓扑并行度很高并执行全对全混洗（重新平衡），默认的 10% 可能不够。增加网络内存的最小/最大限制可以缓解此瓶颈。

### 任务槽与资源隔离

TaskManager 在**任务槽**中执行任务。一个槽位表示 TaskManager 资源的固定部分。值得注意的是，槽位在逻辑层面严格隔离托管内存，但它们共享 TCP 连接和 JVM 堆。这意味着一个槽位中的内存泄漏可能导致整个 TaskManager 崩溃，影响在该节点上运行的所有其他槽位。

槽位分配策略显著影响效率。Flink 默认采用**槽位共享**。这允许数据管道中每个算子的一个子任务共享一个槽位。例如，`Source -> Map -> Sink` 的数据管道可以完全在一个槽位中运行。



![槽位共享效率](plots/8132-0.json)



> 比较独立算子与启用槽位共享时的资源利用开销。

槽位共享提高了资源利用率，因为简单的算子（如 `map`）不需要与负载大的算子（如 `window`）相同的资源。通过将它们共同放置，`map` 操作实质上“免费”使用了为 `window` 分配的资源。

不过，在高级调优中，您可能希望使用**槽位共享组**来打破这种关联。如果您有一个特别负载大的算子，例如复杂的机器学习 (machine learning)推理 (inference)模型，将其与高吞吐量 (throughput)源共同放置可能会导致资源竞争。您可以将负载大的算子隔离到其自己的组中：

```java
// 隔离推理算子
stream.map(new InferenceFunction())
      .slotSharingGroup("gpu-intensive-group")
      .name("InferenceNode");
```

这会强制 Flink 将 `InferenceNode` 放置到不同的槽位中，如果资源允许，甚至可能放置在不同的 TaskManager 上，以确保繁重的计算不会耗尽摄取源的 CPU 周期。

### 优化垃圾回收

由于 Flink 使用长时间存在的对象来存储状态，并使用短时间存在的对象进行处理，这给垃圾收集器（GC）带来了独特的压力。默认的 G1GC 收集器通常是有效的，但对延迟敏感的应用程序需要进行调优。

频繁长时间的 GC 暂停（Stop-The-World 事件）会导致 Flink 错过心跳间隔，从而引发错误的故障检测和作业重启。这通常是任务堆过小或内存泄漏的体现。

为了排查此问题，请监控 `Status.JVM.GarbageCollector.G1_Young_Generation.Time` 指标。如果此指标与吞吐量 (throughput)下降同时出现峰值，请考虑以下调整：

1. **增加任务堆：** 如果您的状态较小但每个事件的对象创建量很大，请将内存从托管内存转移到堆中。
2. **对象复用：** 在 Flink 执行配置中启用对象复用。这会指示 Flink 重用可变对象进行反序列化，而不是为每个记录创建新实例。

   ```java
   ExecutionConfig config = env.getConfig();
   config.enableObjectReuse();
   ```

   *警告：* 仅当您的下游算子立即使用数据或复制数据时，才启用对象复用。如果某个算子持有对传入对象（例如，在窗口缓冲区中）的引用而未进行复制，则数据将被下一个事件覆盖，导致数据损坏。

在 Flink 中配置内存是在堆（用于 Java 逻辑）、托管内存（用于 RocksDB 状态）和网络缓冲区（用于数据流）之间取得平衡。默认配置侧重于安全性，但特定高负载情况需要手动干预以最大化硬件容量。

## 参考资料

- [Flink Memory Configuration](https://nightlies.apache.org/flink/flink-docs-release-1.19/docs/deployment/config/#memory-configuration) — The Apache Flink Community (2024)
  解释了详细的Flink内存模型、TaskManager内存组件以及托管内存和网络缓冲区等配置参数。
- [Task Slots & Resource Management](https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/concepts/flink-architecture/#task-slots-and-resources) — The Apache Flink Community (2023)
  Publisher: Apache Flink
  描述了TaskManager槽位、资源隔离、槽位共享以及用于高效资源利用的槽位共享组。
- [Java Platform, Standard Edition HotSpot Virtual Machine Garbage Collection Tuning Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCK9gXbtmOWYCGhtAMszp2YK19u29jGqZZjdon2CBt3U5QPCLNjCxBn0oGmCoZ1NnFl-a21NO0J4poZLkqZV8PG9hOmVnVsYUNgdE6j96L-UIwqvq1fq-7BrJ0Px1pLw1vz7SrkCl1VkHSqVo9iNsijdL22cnYMg==) — Oracle (2023)
  提供了有关JVM垃圾收集器（包括G1GC）的详细信息，以及针对Java应用的特定调整。
- [Stream Processing with Apache Flink: Fundamentals, Streamlining Dataflow, and Event-Driven Applications](https://www.amazon.com/Stream-Processing-Apache-Flink-Fundamentals/dp/149197428X) — Fabian Hueske, Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  提供了Flink架构的广泛内容，包括内存管理、状态后端和生产优化等部分。
