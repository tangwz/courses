---
course: "building-scalable-data-warehouses"
chapter: "high-throughput-ingestion"
lesson: "late-arriving-data"
sourceId: 8034
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-3-high-throughput-ingestion/late-arriving-data"
title: "处理延迟到达的数据"
description: "处理乱序事件和水位线管理的策略。"
order: 4
plots: ["plots/8034-0.json"]
sourceHash: "cfc2a572a580ac3eed106e6f45c2f8c9f673db429f620a274923849b7f36a3e6"
sourceCorrections: []
---

分布式数据摄取面临时间很少是线性序列的难题。网络分区、设备连接问题和批处理延迟常常导致事件到达数据仓库的时间比实际发生时间晚很多。一个高吞吐量 (throughput)的数据管道必须区分两个不同的时间维度：**事件时间**（事件实际发生的时间）和**处理时间**（系统接收到事件的时间）。

仅依赖处理时间进行分析会引入明显的偏差。如果一笔发生在23:55的交易在次日00:05才被摄取，那么按处理时间戳分组会导致收入被归到错误的财日。反之，严格按事件时间排序需要一套方案来应对那些在分析窗口理论上已经关闭后才到达的数据。

### 时间偏差与水位线

处理时间 $t_p$ 与事件时间 $t_e$ 的差异定义为偏差 $S$：

$S(e) = t_p(e) - t_e(e)$

在一个理想系统中，$S(e)$ 接近零。实际上，$S(e)$ 是一个具有长尾分布的随机变量。为了控制这一点，我们使用**水位线**。水位线是一个启发式函数 $W(t)$，它表明在特定处理时间，系统预计不会再收到时间戳早于 $T$ 的事件。

当水位线通过特定时间戳时，摄取引擎可以关闭窗口并具体化处理结果。任何以 $t_e < W(t_p)$ 到达的数据都被归类为**延迟到达数据**。

> 下图展示了事件时间与处理时间的关系。“理想”线表示零延迟。“水位线”指示系统对延迟的容忍度。水位线以下的点正常处理，而水位线以上的点则视为延迟，并触发特定的处理逻辑。



![事件时间与处理时间](plots/8034-0.json)



### 处理延迟到达的策略

当事件违反水位线限制时，摄取管道必须根据业务要求和重新计算的成本执行三种策略之一。

#### 1. 丢弃

最直接的方法是丢弃在窗口关闭后到达的数据。这在监控系统中很常见，因为实时运行状况比100%的历史准确性更有价值。如果10分钟前的CPU指标现在才到达，它可能已不再具有实际意义。

#### 2. 旁路输出（死信队列）

延迟数据被分流到一个单独的存储位置，通常是“冷”存储桶或特定的 `late_arrivals` 表。这可以避免主管道停滞或重新触发昂贵的计算。工程师可以定期（例如，每晚）运行批处理作业，将这些延迟记录合并到主数据仓库表中。这种方法兼顾了实时稳定性与最终一致性的需求。

#### 3. 更新与重新计算

在金融或监管背景下，数据完整性是强制性的。当延迟数据到达时，系统必须更新之前已具体化的聚合数据。在MPP数据仓库中，这对存储微分区有具体影响。

如果你按日期对数据进行分区，一个 `2023-10-01` 的记录在 `2023-11-01` 到达，会迫使数据仓库解压缩10月份的分区，写入新记录，并重新压缩微分区。此操作是I/O密集型的。为减轻性能影响，工程师常使用**回溯窗口**。

回溯窗口限制系统会向后检查更新的范围。例如，一个在 $H$ 小时运行的摄取作业可能会查询 $[H-1, H]$ 窗口的源数据，同时也会检查 $[H-24, H]$ 范围内记录的更新。

> 下图描绘了一个根据时间限制和分区影响来路由摄取数据的决策流程。

### 双时间建模

为了审计我们*知道什么*以及*何时知道*，高级数据仓库会实施双时间建模。这包括为每条记录存储事件时间与摄取时间。

考虑一个用户更新地址的例子。搬家发生在周一（事件时间），但系统在周三才收到更新（摄取时间）。通过同时存储这两个时间，你可以重构数据库在周二时的状态（显示旧地址）或查询当前真实情况（显示新地址）。

在Snowflake或BigQuery中，这通过向所有表添加 `_ingestion_timestamp` 列并将其用作二级集群键来实现。这使得过滤摄取窗口以进行增量加载的查询更加高效，而分析查询则可以通过过滤事件时间来实现业务逻辑。

### Lambda与Kappa的权衡

延迟数据的处理常决定Lambda和Kappa架构的选择。

- **Lambda架构：** 使用一个速度层进行实时视图，以及一个批处理层进行全面处理。延迟数据由批处理层自然处理，该层定期重新处理整个数据集。这种方法有效，但需要维护两套代码库。
- **Kappa架构：** 将所有内容都视为数据流。延迟数据通过重新处理数据流或使用“撤回”（发出负值以取消先前结果）来处理。这种方法操作上更简单，但需要一个能够长时间管理复杂状态的数据流处理引擎。

在现代MPP数据仓库中，这种区别变得模糊。凭借动态表（Snowflake）或物化视图（BigQuery）等功能，数据仓库实际上充当了一个可以处理微批次更新的数据流引擎。当延迟数据进入底层源表时，只要刷新逻辑定义正确，引擎就会自动处理“撤回”或更新逻辑。

高效管理延迟到达的数据需要精确定义延迟要求。如果业务相关方要求在5分钟内达到99.9%的准确性，那么处理乱序事件的基础设施成本将呈指数级增长。反之，接受24小时的对账期则允许对历史分区进行成本效益高的批处理修复。

## 参考资料

- [The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Unified Stream Processing](https://www.vldb.org/pvldb/vol10/p1753-akidau.pdf) — Tyler Akidau, Alex Balikov, Kaya Bekiroğlu, Slava Chernyak, Josh Haberman, Reuven Lax, Sam McVeety, Daniel Mills, Fraser Nevard, Oleksandr Shvets, Sam Whittle (2017)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 10; Pages: 1753-1766; DOI: [10.14778/3137628.3137637](https://doi.org/10.14778/3137628.3137637)
  本文介绍了数据流模型，该模型为批处理和流处理提供统一方法，定义了事件时间、处理时间及水位线，用于管理时间挑战。
- [Big Data: Principles and Best Practices of Scalable Realtime Data Systems](https://www.manning.com/books/big-data) — Nathan Marz and James Warren (2015)
  Publisher: Manning Publications
  介绍了Lambda架构，一种处理数据的双层方法，包括通过其批处理层处理延迟事件，并将其与Kappa架构进行对比。
- [Bitemporal Data: Theory and Practice](https://www.vitalsource.com/products/bitemporal-data-theory-and-practice-tom-johnston-v9780124080676) — Tom Johnston (2014)
  Publisher: Morgan Kaufmann; Pages: 400
  一本关于双时态数据建模的综合书籍，解释了如何存储和查询随时间变化的数据，包括事件时间和记录时间。
