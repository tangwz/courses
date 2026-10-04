---
course: "intro-data-lake-architectures"
chapter: "file-formats-and-optimization"
lesson: "compression-algorithms"
sourceId: 8002
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-2-file-formats-and-optimization/compression-algorithms"
title: "压缩算法"
description: "根据CPU成本与存储节省情况选择正确的压缩编解码器。"
order: 4
plots: ["plots/8002-0.json"]
sourceHash: "b98639a203d70a5e08476920b0e255f26d627009f1c00342d3ff9ea25b284c7e"
sourceCorrections: []
---

数据湖环境中的压缩不仅仅是为了节省磁盘空间。虽然降低存储成本是一个主要优势，但压缩对I/O吞吐量 (throughput)和查询延迟的影响通常更为明显。当Apache Spark或Trino等分布式查询引擎从对象存储读取数据时，操作涉及三个不同的阶段：从网络读取字节，在内存中解压这些字节，以及将数据解析成可用格式。

如果网络带宽无限，未压缩文件从磁盘传输到CPU的速度会更快，但云网络有其限制。相反，高度压缩的文件减少了网络传输时间，但解压需要大量的CPU周期。选择压缩算法的目标是平衡这些对立的因素。您旨在使总时间 $T_{total}$ 最小化，定义如下：

$T_{total} = \frac{\text{大小}_{\text{压缩}}}{\text{带宽}_{\text{网络}}} + \frac{\text{大小}_{\text{原始}}}{\text{速度}_{\text{解压}}}$

如果CPU解压数据所需时间长于网络传输原始字节所需时间，则压缩就成了瓶颈。

### 可拆分性和并行性

在查看具体算法之前，我们必须处理压缩与分布式处理之间的相互作用。数据湖依赖于将大文件拆分成更小块（分片）的能力，这些分片可以由不同的工作节点并行处理。

某些压缩格式，例如应用于CSV文件的标准Gzip流，是不可拆分的。为了读取`data.csv.gz`文件中的最后一个记录，引擎必须从头开始解压整个流以定位记录边界。这迫使单个CPU核心处理整个文件，从而抵消了分布式计算的优势。

然而，当使用Apache Parquet或ORC等列式格式时，这种限制会改变。这些格式将数据分成内部的“行组”或块。压缩编解码器独立应用于每个块，而不是整个文件。因此，使用Gzip压缩的Parquet文件是完全可拆分的。查询引擎读取文件元数据，识别每个行组的字节范围，并将不同的组分配给不同的工作节点。

### 数据湖中的常见算法

我们主要关注现代数据架构中普遍使用的三种编解码器：Snappy、Gzip和Zstandard (Zstd)。

#### Snappy

Snappy为速度而设计。它不追求最大压缩比，但侧重于极高的解压速度，在现代处理器上常超过500 MB/秒。它以极小的CPU开销提供合理的尺寸缩减（通常为1.5倍到2倍）。

基于这种平衡，Snappy是Apache Parquet的默认压缩编解码器。它适合于“热”数据层，在这些层中，查询延迟是最优先事项。在用户期望几秒内得到响应的交互式分析中，Snappy的低CPU成本确保处理器将其时间用于执行查询逻辑，而不是解压字节。

#### Gzip

基于DEFLATE的Gzip提供高压缩比（文本数据通常为3倍到4倍），但在压缩和解压过程中产生高CPU成本。在数据湖中，Gzip最适合“冷”数据或归档数据，其中存储成本是首要关注点，且访问不频繁。

当与JSON或CSV等文本格式一起使用时，Gzip会创建不可拆分的文件。应避免生成大型（多吉字节）Gzip压缩的JSON文件。相反，如果您必须将Gzip与文本一起使用，请将文件大小调整为大致等于您的目标分区大小（例如128MB - 256MB），以便内部拆分不足不会妨碍并行性。

#### Zstandard (Zstd)

Zstandard是Meta公司开发的一种现代算法，提供与Gzip相当的压缩比，但解压速度更接近Snappy。它具有可调节的压缩级别（范围从负值（追求速度）到22（最大压缩））。

Zstd已成为通用数据湖存储的标准建议。它相对于Gzip提供了“帕累托改进”，在标准设置下，其速度和压缩比都优于Gzip。许多组织正将其Silver和Gold表迁移到Zstd压缩的Parquet，以降低存储成本，同时不牺牲查询性能。



![压缩编解码器性能权衡](plots/8002-0.json)



> 典型解压速度与压缩比的比较。Zstd占据有利的中间位置，兼具高压缩比和高速度。

### 列式特定压缩

列式文件格式的一个独特优势是能够对不同列应用不同的编码和压缩策略。由于列中的数据是统一的（例如，全部是整数或全部是时间戳），压缩算法可以运用这种同质性。

例如，包含低基数字符串数据（如`country_code`）的列与字典编码结合游程编码 (RLE) 效果特别好，之后再进行通用压缩。这种处理流程通常会使存储占用显著小于行式存储的同等数据。

配置数据摄取作业（使用Spark或Flink）时，通常在表或文件级别定义压缩编解码器。然而，底层写入库（如`parquet-mr`）会根据数据类型和统计信息自动处理编码的应用。

### 选择策略

选择正确的算法取决于数据生命周期阶段：

1. **暂存区 / 铜层：** 数据通常以原始JSON或CSV格式到达。如果是归档日志，请使用Gzip或Zstd以最小化S3存储成本。如果文件仅仅是用于处理的暂存区，可拆分性在这里并非主要顾虑。
2. **精选区 / 银层与金层：** 这些层用于支持分析查询。使用Parquet或Iceberg表。
   - **默认选择：** Snappy。它支持度高，速度快，无需调优。
   - **成本优化：** Zstd。如果您的数据湖增长到PB级别，从Snappy切换到Zstd可以节省30%的存储费用。确保您的查询引擎（例如，旧版本的Hive或Presto）支持Zstd编解码器。

以下逻辑流程概述了根据文件格式和访问模式选择编解码器的决策过程。

> 基于文件格式和性能要求选择压缩算法的决策树。

通过正确地将压缩算法与存储格式和查询要求对齐 (alignment)，您可以避免CPU成为一个本应是I/O限制型的架构中的瓶颈。

## 参考资料

- [Apache Parquet Format](https://parquet.apache.org/documentation/) — Apache Software Foundation (2025)
  Publisher: Apache Software Foundation
  描述了 Apache Parquet 列式文件格式的官方规范，详细说明了其结构以及压缩编解码器在可拆分性和并行处理方面的集成。
- [Zstandard: Fast Real-time Compression Algorithm](https://www.usenix.org/conference/atc16/technical-sessions) — Yann Collet, Pascal Massimino, Jean-Marc Odobez (2016)
  Journal: USENIX Annual Technical Conference (ATC); Publisher: USENIX Association; Pages: 249-264; DOI: [10.5555/3008035.3008044](https://doi.org/10.5555/3008035.3008044)
  介绍了 Zstandard 压缩算法，详细说明了其设计旨在实现高压缩比和快速解压速度，相较于以前的编解码器有显著改进。
- [Snappy: A Fast Compressor/Decompressor](https://github.com/google/snappy) — Google (2024)
  Google Snappy 的官方项目存储库，描述了其设计原则，即高压缩和解压速度与实用的压缩比相结合，适用于交互式数据处理。
- [Designing Data-Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于设计可伸缩数据系统基本原则的综合书籍，涵盖了与数据湖优化相关的数据存储、处理和分布式架构。
