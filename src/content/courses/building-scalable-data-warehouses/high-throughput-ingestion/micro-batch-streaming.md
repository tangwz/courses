---
course: "building-scalable-data-warehouses"
chapter: "high-throughput-ingestion"
lesson: "micro-batch-streaming"
sourceId: 8033
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-3-high-throughput-ingestion/micro-batch-streaming"
title: "微批处理与流式摄取"
description: "数据仓库中连续流式处理与微批加载的权衡"
order: 3
plots: ["plots/8033-0.json"]
sourceHash: "dde83e0d35d95c7623ddcd0ba048398615bda2a471277c75a4b0d0607600cc1f"
sourceCorrections: []
---

以数据生成的速度摄取数据通常是架构上的愿景，而非严格的业务要求。可靠的变更数据捕获 (CDC) 是一种捕获源数据修改的方法。选择何种机制将这些捕获到的变更传递到数据仓库，决定了整个平台的效率。MPP 系统中的核心矛盾在于数据新鲜度（延迟）与存储效率（吞吐量 (throughput)）之间。

在传统事务处理系统 (OLTP) 中，单独插入行是标准做法。然而，现代数据仓库使用列式存储格式。这些系统针对读密集型分析工作负载进行了优化，并使用高压缩和元数据剪枝技术。写入小文件或单行会产生一种被称为“小文件问题”的现象，即管理数百万个微小文件的元数据开销超过了数据本身的处理时间。

### 摄取延迟的经济考量

微批处理与流式处理的选择本质上是关于计算资源和存储健康状况的经济决策。当数据生成与数据可用之间的时间减少时，摄取成本会非线性上升。

流式摄取架构尝试立即提供数据，通常在几秒或几毫秒内。微批处理架构将记录累积到缓冲区中，并以固定的时间间隔批量加载，通常时间范围是 5 到 60 分钟。

我们可以将摄取成本函数 $C(l)$ 相对于延迟 $l$ 大致建模为一种反比关系：

$C(l) \propto \frac{V}{l} + K$

其中 $V$ 是数据量，$K$ 代表固定的基础设施开销。当 $l$ 趋近于零时，系统必须让计算资源持续活跃以监听传入事件，从而无法实现使云数据仓库具有成本效益的资源自动暂停或缩减。



![摄取成本与数据延迟](plots/8033-0.json)



> 当延迟要求收紧到 5 分钟以下时，成本呈指数级增长。虚线描绘了对存储优化的不利影响，比如压缩率降低和分区数量增加。

### 微批处理架构

微批处理是高吞吐量 (throughput)数据仓库的默认模式，通常也是最有效的。它与 MPP 系统的特点一致，MPP 系统偏好批量操作。在此模式中，编排层（如 Airflow 或 Dagster）或摄取工具在发出批量加载命令（比如 `COPY INTO`）之前，会将数据累积到对象存储（S3、GCS、Azure Blob）中。

微批处理的主要优势是幂等性和可观察性。如果批次失败，整个文件可以重新处理。另外，创建更大的文件使得数据仓库能够更有效地压缩列式块。

要实现高效的微批处理，您必须调整两个参数 (parameter)：

1. **时间窗口 ($t$):** 刷新缓冲区前等待的最长时间。
2. **大小阈值 ($s$):** 达到最大文件大小（比如 100MB - 250MB 压缩后）后刷新。

加载的触发条件变为：

$Trigger = (T_{current} - T_{last\_load} \geq t) \lor (Size_{buffer} \geq s)$

这种方法可避免在流量低谷时出现小批次，同时确保在需求高峰期缓冲区不会溢出。

#### 文件压缩处理

即使使用微批处理，如果批次过于频繁，文件大小最终可能不尽理想。一种常见做法是在后台执行“压缩”或“清理”。不过，现代平台如 Snowflake 和 BigQuery 现在已大部分自动处理此问题，前提是摄取文件不是极小（比如避免 1KB 文件）。

### 流式摄取实现

数据仓库中的真正流式处理并非指对每个事件执行 `INSERT INTO table VALUES (...)`。这种方法会锁定表元数据并产生严重的争用。相反，现代平台提供专门的流式 API（比如 Snowflake Snowpipe Streaming, BigQuery Storage Write API）。

这些 API 与标准 SQL 插入不同。它们通常写入面向行的预写日志 (WAL) 或针对高并发写入优化的临时缓冲区。之后，由供应商明确管理的后台进程会异步地将这些数据从缓冲区迁移到优化的列式存储中。

> 数据从事件总线流向数据仓库内专门的行式存储缓冲区。后台合并器异步地将这些行转换成优化的列式微分区，以保持读取性能。

#### 精准一次流式处理的挑战

在流式处理中实现幂等性比在微批处理中困难得多。在基于文件的微批处理中，文件名可作为天然的去重标识。在流式处理中，您通常依赖于**偏移量追踪**。

使用流式 API 时，应用或连接器必须追踪最后成功提交记录的偏移量。如果流中断，生产者会有效地“回溯”到最后提交的偏移量。但如果提交确认因网络故障而丢失，生产者可能会重新发送已写入的数据，从而导致重复。

为缓解此问题，先进的摄取设计会在目标表中使用“去重窗口”，或依赖确定性主键来合并更新，尽管这会增加读取端的计算开销。

### 决策矩阵：模式选择

在设计您的数据管道时，请使用以下标准来区分流式处理的必要性与微批处理的适用性。

**如果满足以下条件，请选择微批处理（15-60 分钟）：**

- 数据主要用于每日或每小时报告。
- 成本控制是首要考虑因素。
- 转换逻辑涉及需要稳定数据快照的复杂连接。
- 您需要通过文件重新加载来简单重放历史数据。

**如果满足以下条件，请选择微批处理（1-5 分钟）：**

- 业务团队需要“近实时”仪表板，但能容忍轻微延迟。
- 您希望避免管理专用流式基础设施（比如 Kafka Connect 集群）的复杂性。
- 该平台允许频繁加载文件，而不会产生巨大的元数据开销。

**如果满足以下条件，请选择流式处理（< 1 分钟）：**

- 用例涉及运营监控、欺诈检测或实时库存管理。
- 下游应用会立即基于传入事件触发自动化操作。
- 工程团队有能力管理偏移量监控以及失败事件的死信队列 (DLQs)。

在本课程中，我们强调 **流式处理不应成为默认选项**。它是针对特定高价值、低延迟数据集的优化方案。对于大多数分析工作负载，微批处理管道在稳定性、成本和性能之间提供最佳平衡。

### 流中模式演变的应对

流式摄取中一个经常被忽视的重要方面是模式演变。在批处理过程中，如果源中添加了列，批次加载可能会失败，从而提醒工程师更新模式。影响仅限于该批次。

在连续流中，模式不匹配可能会污染管道或导致消费者完全丢弃消息。为应对此问题，高吞吐量 (throughput)管道通常首先将数据摄取到 `VARIANT` 或 `JSON` 列类型中（半结构化数据）。这使得管道无论模式如何变化都能成功运行。数据的结构化和类型化随后被推迟到下游视图或转换过程，这是一种通常被称为“读时模式”的做法。这种方法将摄取基础设施的稳定性与应用数据模型的波动性分离开来。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  介绍批处理和流处理、分布式系统、一致性模型和数据存储的核心概念，为理解数据摄取挑战提供了全面背景。
- [C-Store: A Column-Oriented DBMS](http://www.vldb.org/archives/website/2005/program/paper/thu/p553-stonebraker.pdf) — Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cherniack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel Madden, Elizabeth J. O'Neil, Patrick E. O'Neil, Alexander Rasin, Nga Tran, Stanley B. Zdonik (2005)
  Journal: Proceedings of the 31st International Conference on Very Large Data Bases (VLDB); Publisher: ACM; Pages: 553-564; DOI: [10.14778/1083592.1083658](https://doi.org/10.14778/1083592.1083658)
  介绍了列式数据库管理系统的设计原则，这些系统是现代分析数据仓库的基础，并解释了其针对读密集型工作负载的优化以及对写入操作的影响。
- [Snowpipe Streaming API](https://docs.snowflake.com/en/user-guide/data-ingestion-streaming) — Snowflake (2024)
  Publisher: Snowflake
  提供了领先的云数据仓库如何将低延迟、高吞吐量流式摄取实现到列式存储中的实际细节，讨论了所描述的面向行的预写日志（WAL）和后台合并器。
- [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying-abstraction) — Jay Kreps (2013)
  Publisher: LinkedIn Engineering Blog
  阐述了事务日志和变更数据捕获 (CDC) 在现代数据系统中的基础作用，与章节中“可靠的变更数据捕获 (CDC)”概念直接相关。
