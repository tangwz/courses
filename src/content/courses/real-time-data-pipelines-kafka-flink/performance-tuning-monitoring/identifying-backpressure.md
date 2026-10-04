---
course: "real-time-data-pipelines-kafka-flink"
chapter: "performance-tuning-monitoring"
lesson: "identifying-backpressure"
sourceId: 8131
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-7-performance-tuning-monitoring/identifying-backpressure"
title: "识别反压"
description: "使用 Flink 指标和 UI 定位并诊断反压来源。"
order: 1
plots: ["plots/8131-0.json"]
sourceHash: "e356655ca9dc909f6398e1529b8a6b80dc5ba94a1e614f55fb72f3028c39c892"
sourceCorrections: []
---

反压常被误认为是系统故障。反压在流式架构中作为一种必需的流量控制机制，让系统能够在负载高峰期间保持稳定。它通过严格限制传输中的数据量，防止进程过载。当下游算子无法跟上摄取速率时，它会拒绝新的数据缓冲区。这种拒绝会通过网络栈向上游传播，最终减慢数据源的速度。尽管此机制保持了系统稳定，但它预示着需要立即调整的性能瓶颈。

对数据工程师来说，目标并非在瞬时高峰期完全消除反压，而是要识别违反延迟服务等级协议 (SLA) 的长期存在的反压。在 Apache Flink 中，反压是通过网络栈内的基于信用的流量控制系统实现的。理解此内部机制对于正确解读指标是必需的。

### 基于信用的流量控制

Flink 使用网络缓冲区在任务（运行在不同的 TaskManager 上）之间传输数据。在 Flink 1.5 之前，这高度依赖于 TCP 流量控制。现代 Flink 版本在应用层面采用基于信用的机制，以达到更高的吞吐量 (throughput)和更低的延迟。

在此模型中，接收任务（下游）向发送任务（上游）授予“信用”。一个信用对应一个可用的网络缓冲区。上游任务只有在拥有信用时才能发送数据缓冲区。数据发送后，信用即被消耗。如果下游任务处理落后，其缓冲区将填满，并停止发出信用。结果是，上游任务等待，有效限制了其处理速率。

> TaskManager 之间数据和信用的流动。当下游输入门因缓冲区满而停止授予信用时，上游算子将被阻塞。

此机制将 Flink 的流量控制与底层 TCP 连接解耦，防止单个慢速任务阻塞所有共享同一物理连接的其他任务。

### 反压指标分析

要诊断管道停滞的位置，您必须查看 `backPressuredTimeMsPerSecond` 指标。此指标衡量任务等待网络缓冲区（信用）可用所花费的时间。

Flink 在 Web UI 中总结此状态如下：

- **正常**: 0% ≤ 反压比 ≤ 10%
- **低**: 10% < 反压比 ≤ 50%
- **高**: 50% < 反压比 ≤ 100%

诊断的简单方法是寻找标记 (token)为“高”（红色）的任务。然而，出现反压的任务很少是根本原因。如果任务 A 报告高反压，这意味着任务 A 运行正常但无法将数据推送到任务 B。任务 B 才是瓶颈（或与任务 B 的网络连接）。

#### 定位瓶颈

您通过沿着流图从源到汇追踪反压。您要寻找反压指标从高变为低/正常的状态转换点。

1. **高反压**: 任务空闲，因为它正在等待下游信用。
2. **低/正常反压**: 任务要么处理数据缓慢（CPU 密集型），要么等待外部 I/O（I/O 密集型），或者根本没有数据可处理。

如果任务 A 状态为高，任务 B 为低，那么任务 B 就是限制因素。任务 B 忙于处理，无法足够快地回收缓冲区以授予信用给 A。



![反压传播分析](plots/8131-0.json)



> 拓扑中的反压指标。“窗口聚合”算子是瓶颈，这可以从其之前“KeyBy”算子处反压急剧下降来识别。

在上述场景中，源、映射和 KeyBy 算子都为红色（高）。它们运行正常但受到限制。窗口聚合是绿色（低）。这表明窗口算子是实际瓶颈。它正在尽可能快地处理数据，但慢于上游速率。汇也为绿色，因为它缺乏数据。

### 利用率指标

除了高级别的反压状态之外，特定的缓冲区利用率指标提供细致的调整信息。

- `outPoolUsage`: 表明输出缓冲区的充满程度。如果此值较低，同时反压较高，则证实问题严格在于下游信用。
- `inPoolUsage`: 表明输入缓冲区的充满程度。持续接近 100% 的值证实该算子是瓶颈。

如果您观察到 `inPoolUsage` 较低但上游存在反压，问题可能不是 CPU 或逻辑。这可能是**数据倾斜**。在倾斜场景中，一个特定的并行子任务（例如子任务 3）过载，而其他子任务（子任务 0、1、2）空闲。Flink UI 聚合这些指标。您必须展开算子视图以检查各个子任务。如果只有一个子任务显示 100% 的 `inPoolUsage` 和高繁忙时间，则重新分区数据是必需的。

### 对检查点的影响

反压不仅增加延迟；它还会危及容错性。Flink 的检查点屏障随数据流流动。如果数据流因反压而停滞，检查点屏障也会停滞。

检查点持续时间定义为：

\text{持续时间} = \text{T\_{最后确认}} - \text{T\_{触发}}

在有反压的管道中，屏障从源到汇缓慢移动。这会导致：

1. **检查点超时**: 屏障未能在配置的超时窗口内到达汇。
2. **状态大小膨胀**: 如果禁用未对齐 (alignment)检查点，算子必须在等待所有输入通道的屏障对齐时缓冲数据，从而消耗过多内存。

在诊断检查点故障时，始终首先验证反压状态。解决流量瓶颈通常是稳定检查点机制的唯一方法。

## 参考资料

- [Apache Flink Documentation: The Network Stack](https://flink.apache.org/docs/1.19/docs/ops/production/network_stack/) — The Apache Flink Community (2024)
  Publisher: The Apache Flink Community
  详细解释了Flink的网络栈，包括基于信用的流控制机制及其在管理数据传输和背压方面的作用。
- [Stream Processing with Apache Flink: Fundamentals, StreamSQL, and Table API](https://www.oreilly.com/library/view/stream-processing-with/9781492049962/) — Alexander Walden, Kostas Tzoumas (2020)
  Publisher: O'Reilly Media
  一本关于Apache Flink的综合书籍，提供了对其架构、性能调优策略和操作方面的见解，包括流控制和背压的讨论。
