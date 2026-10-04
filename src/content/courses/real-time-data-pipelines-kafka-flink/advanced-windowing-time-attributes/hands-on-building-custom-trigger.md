---
course: "real-time-data-pipelines-kafka-flink"
chapter: "advanced-windowing-time-attributes"
lesson: "hands-on-building-custom-trigger"
sourceId: 8119
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-4-advanced-windowing-time-attributes/hands-on-building-custom-trigger"
title: "动手实践：制作自定义触发器"
description: "制作自定义触发器，根据外部信号或数据模式触发窗口计算。"
order: 5
plots: ["plots/8119-0.json"]
sourceHash: "1ac313053dfe39af2698c0e9586db985112840f829bd0a19007e61c2fca4175c"
sourceCorrections: []
---

DataStream API 提供的标准触发器可应对处理时间间隔或元素计数等常见用例。但是，在实际生产中，数据处理管道经常会遇到这样的情况：窗口计算必须依据复杂的、特定业务的逻辑，而不是简单的时钟周期或计数。例如，您可能需要提早发出窗口结果，因为检测到特定的“毒丸”模式，或者您可能需要实现一种动态超时机制，该机制会根据数据流当前吞吐量 (throughput)进行调整。

在本节实际操作中，我们将设计一种实时 AI 管道中常用的混合触发机制：**时间感知批处理触发器**。这种模式对于低延迟推理 (inference)服务很有用，在此类服务中，您需要累积一批记录（向量 (vector)）以发送到模型端点。您不能无限期等待批次填满，也不能在吞吐量允许高效批处理时发送单个记录。此触发器将在计数达到阈值或特定超时时间经过时触发窗口计算，以先发生者为准。

### 触发器接口

实现自定义触发器需要扩展抽象的 `Trigger` 类。Flink 运行时会根据不同信号调用此类上的特定方法。理解这些方法的生命周期对于正确的状态管理很必要。

您必须实现的核心方法是：

1. **onElement:** 当每个元素到达窗口时调用。
2. **onProcessingTime:** 当注册的处理时间计时器触发时调用。
3. **onEventTime:** 当注册的事件时间计时器触发时调用。
4. **clear:** 当窗口被清理时调用，以清除任何上下文 (context)状态。

每个方法都会返回一个 `TriggerResult`，它指示窗口操作符应采取的行动。

- `CONTINUE`: 不执行任何操作。
- `FIRE`: 计算窗口函数（发出结果）但保留窗口状态。
- `PURGE`: 清除窗口内容而不进行计算。
- `FIRE_AND_PURGE`: 计算窗口函数，然后清除内容。

### 时间感知批处理的逻辑流程

我们正式定义触发逻辑。设 $N$ 为最大批次大小，$\Delta t$ 为最大等待时间。对于一个起始于 $t_0$ 且包含元素集 $E$ 的窗口，我们在以下情况触发：

$|E| \ge N \quad \lor \quad \text{当前时间} \ge t_0 + \Delta t$

以下图表概述了 `onElement` 和计时器方法中所需的决策逻辑。

> 混合计数与时间触发器的流控制。系统优先考虑填满批次，但通过计时器确保延迟限制得到遵守。

### 实现触发器

我们在触发器内部使用 Flink 的管理状态来追踪元素计数。由于触发器可以触发多次或在故障后仍能持久运行，因此计数必须存储在 `ReducingState` 或 `ValueState` 中，而不是局部变量里。

下面是实现逻辑。我们使用 `ValueState` 来保存当前计数和已注册计时器的时间戳。这可以确保我们能够对计时器注册进行去重。

```java
public class BatchingTrigger<T, W extends Window> extends Trigger<T, W> {
    private final long maxCount;
    private final long timeoutMillis;

    // 用于访问 Flink 管理状态的状态描述符
    private final ValueStateDescriptor<Long> countStateDesc = 
        new ValueStateDescriptor<>("count", Long.class);
    private final ValueStateDescriptor<Long> timerStateDesc = 
        new ValueStateDescriptor<>("timer", Long.class);

    public BatchingTrigger(long maxCount, long timeoutMillis) {
        this.maxCount = maxCount;
        this.timeoutMillis = timeoutMillis;
    }

    @Override
    public TriggerResult onElement(T element, long timestamp, W window, TriggerContext ctx) throws Exception {
        ValueState<Long> countState = ctx.getPartitionedState(countStateDesc);
        ValueState<Long> timerState = ctx.getPartitionedState(timerStateDesc);

        long currentCount = countState.value() == null ? 0L : countState.value();
        long newCount = currentCount + 1;
        countState.update(newCount);

        // Condition 1: Count threshold reached
        if (newCount >= maxCount) {
            // Clear any pending timer to avoid double firing
            long timerTimestamp = timerState.value() == null ? 0L : timerState.value();
            if (timerTimestamp > 0) {
                ctx.deleteProcessingTimeTimer(timerTimestamp);
                timerState.clear();
            }
            countState.clear();
            return TriggerResult.FIRE_AND_PURGE;
        }

        // Condition 2: Set timer if not already set
        if (timerState.value() == null) {
            long timer = ctx.getCurrentProcessingTime() + timeoutMillis;
            ctx.registerProcessingTimeTimer(timer);
            timerState.update(timer);
        }

        return TriggerResult.CONTINUE;
    }

    @Override
    public TriggerResult onProcessingTime(long time, W window, TriggerContext ctx) throws Exception {
        // Timer fired implies timeout reached before count filled
        ctx.getPartitionedState(countStateDesc).clear();
        ctx.getPartitionedState(timerStateDesc).clear();
        return TriggerResult.FIRE_AND_PURGE;
    }

    @Override
    public TriggerResult onEventTime(long time, W window, TriggerContext ctx) {
        return TriggerResult.CONTINUE;
    }

    @Override
    public void clear(W window, TriggerContext ctx) throws Exception {
        ValueState<Long> timerState = ctx.getPartitionedState(timerStateDesc);
        long timerTimestamp = timerState.value() == null ? 0L : timerState.value();
        
        if (timerTimestamp > 0) {
            ctx.deleteProcessingTimeTimer(timerTimestamp);
        }
        
        ctx.getPartitionedState(countStateDesc).clear();
        timerState.clear();
    }
}
```

### 触发器延迟分析

`maxCount` 和 `timeoutMillis` 的选择在吞吐量 (throughput)（批次效率）和延迟之间产生了一种权衡。一个纯粹依靠计数的触发器可以提供最佳吞吐量，但在低流量期间会产生无限制的延迟。一个纯粹依靠时间的触发器则提供有界延迟，但可能导致批次小且效率低下。

下图展示了这种混合触发器在不同输入速率下的延迟曲线。请注意，即使到达速率大幅下降，延迟也会限制在超时值以内。



![延迟曲线：混合触发器 vs. 固定计数](plots/8119-0.json)



> 固定计数触发器与混合触发器之间的延迟比较。混合方法将最大延迟限制在 200 毫秒，在低吞吐量时有效控制了固定计数策略中出现的“长尾”延迟。

### 状态清理与良好实践

实现自定义触发器时，`clear()` 方法并非可选。当窗口被清理（例如，因允许的迟到过期）或当窗口被另一个触发器结果显式清理时，Flink 会调用 `clear()`。未能清理状态会导致资源泄漏。

在上述实现中，我们专门删除了已注册的处理时间计时器。尽管 Flink 会在窗口过期时自动清理窗口状态，但 `TriggerContext` 中手动注册的计时器会一直存在，除非被显式删除。如果您的数据流生成数百万个窗口，孤立的计时器会大幅降低检查点机制的性能并增加堆内存使用。

### 与窗口 API 集成

要使用 `BatchingTrigger`，您需要将其附加到键控流窗口定义上。这会替换默认触发器（通常是 `EventTimeTrigger`）。

```java
DataStream<Batch> batches = inputStream
    .keyBy(Event::getKey)
    .window(GlobalWindows.create())
    .trigger(new BatchingTrigger<>(10, 200))
    .process(new BatchInferenceFunction());
```

我们通常将 `GlobalWindows` 与此模式结合使用，因为触发器本身管理窗口的生命周期（通过 `FIRE_AND_PURGE`）。其逻辑不依据固定的开始和结束时间，而是依据数据到达模式。这种配置有效地创建了数据驱动的窗口，这些窗口能够适应数据流的速度。

## 参考资料

- [Apache Flink Documentation: Windowing](https://nightlies.apache.org/flink/flink-docs-release-1.19/docs/dev/datastream/operators/windows/) — The Apache Flink Community (2024)
  Publisher: Apache Software Foundation
  Flink窗口API的官方指南，包含如何实现自定义触发器、管理定时器以及处理TriggerResult操作。
- [Stream Processing with Apache Flink: Fundamentals, StreamSQL, and Data-Driven Applications](https://learning.oreilly.com/library/view/stream-processing-with/9781491974281/) — Fabian Hueske and Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  由Flink贡献者编写的资源，详细介绍了Flink的架构、窗口机制以及自定义触发器实现和状态管理等高级特性。
- [Apache Flink Documentation: State & Fault Tolerance](https://nightlies.apache.org/flink/flink-docs-release-1.19/docs/concepts/stateful-stream-processing/) — The Apache Flink Community (2024)
  Publisher: Apache Software Foundation
  解释了Flink的托管状态，包括ValueState和状态清理实践，这对于构建健壮且容错的自定义触发器至关重要。
