---
course: "real-time-data-pipelines-kafka-flink"
chapter: "real-time-ai-feature-engineering"
lesson: "online-feature-generation"
sourceId: 8136
sourceUrl: "https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-8-real-time-ai-feature-engineering/online-feature-generation"
title: "在线特征生成"
description: "实时计算机器学习模型的滑动窗口聚合和复杂特征。"
order: 1
plots: ["plots/8136-0.json"]
sourceHash: "d5025127dfdcbfe4c627ad170c4ce8852413bb4ef2efbf7c7f69bd9495eedef1"
sourceCorrections: []
---

特征工程将原始数据转换为适合机器学习 (machine learning)算法的格式。这通常涉及在批处理中对数据仓库的静态分区执行SQL查询。然而，这些静态分区会引入较长的延迟。当特征向量 (vector)计算出来时，用户情境可能已经改变，使得预测不再相关。在线特征生成将此计算转移到流处理层，在事件抵达时计算特征值，以保证模型输入能反映环境的即时状况。

### 有状态流聚合

在数据流上生成特征的主要方式是有状态聚合。与过滤或映射等无状态转换不同，特征生成需要上下文 (context)。要计算“过去一小时的平均交易价值”或“过去5分钟的点击次数”，处理器必须保存历史记录。

在 Apache Flink 中，我们通过窗口逻辑与键控状态结合来实现这一点。原始流通过 `keyBy` 操作符根据逻辑实体（例如 `user_id` 或 `device_id`）进行分区。这使得特定实体所有事件都被导向到相同的并行任务槽，从而在窗口累积阶段无需网络数据重排即可访问本地状态。

> 窗口聚合的数据流。事件按键分片，对照本地状态进行处理，并作为更新后的特征向量 (vector)发出。

### 增量聚合优化

特征生成中常见的性能瓶颈源于原始事件的缓冲。如果一个窗口持续24小时并包含数百万个事件，在窗口触发前将所有负载存储在状态后端是低效的。这会增加内存压力，并在评估阶段造成CPU使用率峰值。

为了减轻此问题，我们采用增量聚合。Flink 不存储事件，而是更新一个紧凑的累加器状态。对于简单平均值，状态仅包含运行总和与计数。当新事件 $x_t$ 到来时，我们立即更新状态：

$S_t = S_{t-1} + x_t$
$N_t = N_{t-1} + 1$

当窗口关闭或触发时，特征简单地计算为 $S_t / N_t$。这种方式使得存储复杂度保持为 $O(1)$，无论窗口中的事件数量有多少。

在 Flink 中，这通过 `AggregateFunction` 接口来实现。该接口需要定义一个累加器、一个添加方法和一个结果获取方法。`merge` 方法对于会话窗口也不可或缺，其中多个部分聚合可能需要合并。

### 滑动窗口与跳跃性能

滑动窗口在特征工程中普遍存在。一个特征，例如“过去15分钟内的登录失败次数，每分钟更新”，这意味窗口大小为15分钟，滑动步长（跳跃）为1分钟。

虽然简单，但如果未经调整，滑动窗口的计算成本可能很高。单个事件属于 $Size / Slide$ 个窗口。对于每分钟滑动一次的1小时窗口，每个事件属于60个重叠窗口。如果 Flink 为每个窗口桶复制事件，状态大小将急剧增加。

Flink 通过根据窗口和滑动间隔的最大公约数将流切分为“窗格”来优化此过程。然而，对于高吞吐量 (throughput)的特征生成，使用指数移动平均 (EMA) 通常更有效率。EMA 对近期数据点赋予更高的权重 (weight)，并通过数学方式衰减旧信息，从而无需严格的窗口边界和缓冲区管理。

EMA 的递归公式是：

$EMA_t = \alpha \cdot x_t + (1 - \alpha) \cdot EMA_{t-1}$

这里，$\alpha$ 代表平滑因子，其中 $0 < \alpha < 1$。较高的 $\alpha$ 会更快地降低旧观测值的权重。这种计算只需存储前一个 $EMA$ 值，使其在低延迟特征管道中非常高效。

### 处理时间偏差和水印

实时特征很大程度上依赖于时间的准确性。如果网络问题导致事件延迟，它可能在其时间戳所属的窗口理论上关闭之后才抵达。在欺诈识别中，乱序处理交易可能导致特征向量 (vector)遗漏一个必要信号，从而产生误报。

Flink 通过水印处理此问题。水印是一种随流流动并声明不会再有时间戳小于 $T$ 的事件抵达的机制。您必须配置水印生成策略以平衡延迟和完整性。

1. **严格/低延迟：** 水印紧随最大时间戳生成。这会减少特征发出的延迟，但会增加丢弃延迟数据的风险。
2. **包容/高完整性：** 水印显著落后于最大时间戳。这使得延迟事件可以包含在聚合中，但会推迟特征的可用性。

对于在线推断，我们通常不能等待延迟数据。标准的做法是配置一个较短的允许延迟期，并将任何非常延迟的数据重定向到侧输出进行监看，以保证主要特征管道保持低延迟。

### 处理高基数

为数百万用户生成特征时，状态后端成为制约因素。由于垃圾回收暂停，将特征状态存储在Java堆上很少可行。相反，我们配置 RocksDB 作为状态后端。RocksDB 将状态存储在本地磁盘（SSD）上，并依赖堆外内存进行缓存。

然而，RocksDB 序列化会增加开销。每次状态访问都需要将键和值序列化为字节。要改进此情况：

- 为累加器使用原始数组或专门的序列化格式（如 Protobuf），而不是沉重的Java对象。
- 启用增量检查点，以减少将状态快照持久化到分布式存储（S3/HDFS）所需的I/O带宽。
- 关注 `state.backend.rocksdb.block-cache-usage` 指标。如果缓存过小，系统频繁从磁盘获取数据将导致读取放大。

下图说明了当吞吐量 (throughput)增加时，与基于磁盘的状态交互和内存计算之间的延迟权衡。



![不同状态后端下的延迟与吞吐量对比](plots/8136-0.json)



> 延迟特性比较。堆状态提供较低的初始延迟，但在内存压力（GC）下性能迅速下降，而 RocksDB 在较高吞吐量下保持稳定表现。

### 流中的特征编码

除了数值聚合，模型通常还需要分类特征。在批处理系统中，独热编码 (One-Hot Encoding) 或目标编码 (Target Encoding) 等技术是使用数据集的完整词汇表 (vocabulary)来应用的。在流中，词汇表是无界的且不断变化的。

对于独热编码，在状态中维护动态字典是危险的，因为它会无限制增长。一种常见的流友好替代方案是特征哈希（哈希技巧）。我们将分类字符串哈希到一个固定的整数空间。

$Index = hash(category\_string) \pmod{Vector\_Size}$

这种容易发生冲突的方法牺牲准确性以换取恒定的内存占用，这是流环境中必要的折衷。另外，对于目标编码（用目标变量的平均值替换分类），我们使用前面描述的相同增量聚合逻辑：为每个分类键维护目标的运行总和和计数，并实时更新映射。

通过掌握这些聚合和状态管理技术，您可以保证为AI模型服务的特征向量 (vector)既计算高效又具有时间上的相关性。

## 参考资料

- [Stateful Stream Processing](https://flink.apache.org/2020/01/29/state-unlocked-interacting-with-state-in-apache-flink.html) — The Apache Flink Community (2020)
  Publisher: Apache Flink
  官方文档，说明 Flink 的状态、时间机制（窗口、水印）和 RocksDB 等状态后端。
- [Stream Processing with Apache Flink: Fundamentals, StreamSQL, and Table API](https://learning.oreilly.com/library/view/stream-processing-with/9781491974281/) — Fabian Hueske, Vasia Kalavri (2019)
  Publisher: O'Reilly Media
  一本关于 Apache Flink 的全面书籍，涵盖其架构、状态管理和事件时间处理。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供分布式系统、流处理和支持实时应用的数据架构基础概念。
