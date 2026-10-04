---
course: "intro-data-lake-architectures"
chapter: "file-formats-and-optimization"
lesson: "apache-parquet-internals"
sourceId: 8000
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-2-file-formats-and-optimization/apache-parquet-internals"
title: "Apache Parquet 内部机制"
description: "了解 Parquet 文件结构、游程编码和字典压缩，以实现高效存储。"
order: 2
plots: ["plots/8000-0.json"]
sourceHash: "686c4d15baf2b793248b284e38a8f33f69e680ffe3011d19e21bdb26a3681160"
sourceCorrections: []
---

Apache Parquet 已成为数据湖存储的事实标准，因为它解决了基于文本格式的根本性低效问题。虽然 CSV 和 JSON 对人来说易于阅读，但对机器来说，处理它们的计算开销很大。每当查询引擎读取一个 CSV 文件时，它必须解析每个字节，处理分隔符，并推断数据类型。Parquet 颠倒了这种优先次序。它是一种为机器优化的二进制、自描述格式，专门设计用于在分析查询期间减少输入/输出 (I/O) 操作和 CPU 使用。

### Parquet 的层次结构

为了了解 Parquet 如何实现高性能，我们必须查看其内部组织。它不是将数据写入为连续的值流，而是将数据划分为明确的层次结构：文件、行组、列块和页面。

1. **行组 (Row Group)：** 文件被水平分区为行组。一个行组逻辑上包含一组行（例如，10,000 行或 128MB 数据）。这种水平切分使查询引擎能够并行执行操作。一个工作节点可以处理第一个行组，而另一个处理第二个。
2. **列块 (Column Chunk)：** 在一个行组内部，数据被垂直切分。数据集中的每个列都有一个对应的列块。这是数据发生物理分离的地方。如果您只选择 `user_id` 列，引擎将只读取与 `user_id` 相关联的列块，而忽略 `timestamp`、`event_type` 或 `payload` 的块。
3. **页面 (Page)：** 列块被进一步细分为页面。页面是存储的原子单元。压缩和编码应用于页面级别。当数据库引擎读取数据时，它每次取一个页面。

> Parquet 文件的物理布局，展示了数据如何先被分割成行组和列块，然后再被分成页面。

### 文件元数据和统计信息

Parquet 的性能提升很大程度上依赖于元数据。与许多将文件头放在开头的格式不同，Parquet 将文件元数据写入页脚。这种设计选择允许数据写入器在内存中缓冲数据，并将其顺序写入磁盘。一旦所有数据写入完成，写入器会聚合所有行组的位置和统计信息，并将其附加到文件末尾。

元数据包含模式、版本信息以及每个列块的偏移量。更重要的是，它存储了每个列块和页面的统计信息，例如：

- 最小值
- 最大值
- 空值计数

当您执行 `SELECT * FROM orders WHERE order_amount > 500` 这样的查询时，查询引擎会首先读取页脚。它会检查 `order_amount` 列块的元数据。如果某个行组的 `min: 0` 和 `max: 100`，引擎就能确切知道该组中没有行符合筛选条件。它会跳过整个行组，避免了读取或解压该数据的必要。这种技术比扫描每个值明显更快。

### 编码方案

Parquet 通过高效编码减少存储成本。编码不同于压缩（如 Snappy 或 Gzip）。压缩是一种应用于字节流的通用算法，而编码则运用数据类型和分布的知识来更紧凑地表示值。

#### 字典编码

这是许多数据类型的默认编码。它对于低基数列（即唯一值的数量相对于总行数较少，例如国家名称、产品类别或状态标志）非常高效。

Parquet 不是将字符串“United States”存储数千次，而是在列块的开头创建一个字典。

1. “美国” $\rightarrow$ 0
2. “法国” $\rightarrow$ 1
3. “日本” $\rightarrow$ 2

实际数据随后存储为小整数流（索引）。如果列数据是 `["United States", "United States", "France"]`，Parquet 存储 `[0, 0, 1]`。这会将长字符串所需的空间减少到每行几个比特。

#### 游程编码 (RLE)

游程编码 (RLE) 和位封装通常与字典编码结合使用或用于布尔值。RLE 压缩重复数据序列。如果列包含有序或重复数据，RLE 效率极高。

设想一个状态列，其中值“Active”重复 500 次，随后是“Inactive”200 次。

- **原始存储：** 存储“Active”500 次。
- **RLE 存储：** 存储对 `(500, “Active”), (200, “Inactive”)`。

这将 700 个数据点变成了两个元组。当与位封装结合时，整数被压缩到所需的最少位数。例如，如果页面中的最大值是 7，Parquet 每个整数只需 3 个比特 ($2^3 = 8$)，而不是标准的 32 比特。



![按编码类型划分的存储效率 (模拟)](plots/8000-0.json)



> 比较在高度重复的数据集上应用字典编码和 RLE 时存储占用减少的情况。

### 数据类型：物理类型与逻辑类型

Parquet 区分数据如何存储（物理类型）和如何被解释（逻辑类型）。这种解耦使格式保持简洁，同时支持复杂数据结构。

只有少数几种基本物理类型：

- `BOOLEAN`
- `INT32`
- `INT64`
- `INT96` (历史类型，主要用于时间戳)
- `FLOAT`
- `DOUBLE`
- `BYTE_ARRAY`
- `FIXED_LEN_BYTE_ARRAY`

逻辑类型映射到这些基本类型。例如，一个 `UTF8` 字符串物理上存储为 `BYTE_ARRAY`，并带有一个逻辑注解，告诉读取器将这些字节解释为 UTF-8 字符串。一个 `DECIMAL` 值可能根据所需的精度存储为 `FIXED_LEN_BYTE_ARRAY` 或 `INT64`。这种抽象使得 Parquet 能够发展并支持新类型（如 UUID 或时间间隔），而无需更改底层的二进制读取器。

理解这些内部机制有助于设计模式。例如，了解到 `INT32` 是日期的底层存储，表明在将数据写入 Parquet 之前按日期排序数据将最大限度地提高游程编码的效率，从而得到更小的文件和更快的查询。

## 参考资料

- [Apache Parquet Format Specification](https://github.com/apache/parquet-format/blob/master/spec.md) — Apache Parquet Project (2024)
  提供了Apache Parquet文件格式的权威技术规范，详细说明了其内部结构、元数据、编码和数据类型。
- [C-Store: A Column-Oriented DBMS](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGO1LoYITvTzr9iStX9DoxxgFD5BzlTbqpyQwvFykAXMwrJfA1apfBzjbH1UcdU-ZozwHGYS9HOWxYY6PPtEjJfs_a0YyAQdzzIEPfX8-ljmN6qllfbrhSn7YVTsoQs25ULy2ndyJVZe4kMpNbNWorqz_g0gnvG7wc7pTn1ksiCs3oZ7ltjortX1Cth) — Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cherniack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel R. Madden, Elizabeth J. O'Neil, Patrick E. O'Neil, Alexander Rasin, Nga Tran, Stan B. Zdonik (2005)
  Journal: Proceedings of the 31st International Conference on Very Large Data Bases (VLDB); Publisher: VLDB Endowment; Pages: 553-564; DOI: [10.1007/11566322_38](https://doi.org/10.1007/11566322_38)
  描述了列式数据库系统的基本概念，解释了其在分析工作负载中实现效率的原理，这些原理直接应用于Parquet等格式。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media; Pages: 614
  第三章“存储与检索”包含关于列式存储和编码的部分，提供了一个概念框架，有助于理解Parquet在分析工作负载方面的设计选择。
