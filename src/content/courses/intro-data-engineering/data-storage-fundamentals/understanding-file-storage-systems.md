---
course: "intro-data-engineering"
chapter: "data-storage-fundamentals"
lesson: "understanding-file-storage-systems"
sourceId: 5560
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-4-data-storage-fundamentals/understanding-file-storage-systems"
title: "理解文件存储系统"
description: "了解大数据中使用的分布式文件系统，例如HDFS。"
order: 4
plots: []
sourceHash: "859489452797cac9e3a4eae8fbc8f04cd115143bda66ceb05f9ca9230c4dd0a8"
sourceCorrections: []
---

您对个人电脑上的文件系统很熟悉。它们在单个硬盘上组织文件和文件夹。但如果您的数据量变得太大，无法在一台机器上存储，例如达到TB甚至PB级，会发生什么呢？或者如果您需要多台计算机协同工作来快速处理这些海量数据，又该怎么办？这就是 **分布式文件系统 (DFS)** 变得很重要的原因。

分布式文件系统不再将文件存储在一台计算机上，而是将文件分散存储在由多台机器组成的网络中，这些机器通常被称为集群。然而，它向用户和应用程序呈现这些文件时，就好像它们都存储在一个位置一样。这种方法为处理大型数据集提供了几个重要优点：

- **扩展性：** 需要更多存储空间？只需向集群添加更多机器（节点）。系统可以横向扩展以容纳巨量数据。
- **容错性：** 如果集群中的一台机器出现故障，数据不会丢失。系统通常在不同的机器上存储数据的副本（复制品），以确保可用性。
- **高吞吐量 (throughput)：** 通过在多台机器上分发数据和处理，像读取大型文件进行分析这样的任务通常可以并行地更快完成。

### Hadoop 分布式文件系统 (HDFS)

分布式文件系统中最知名的一个例子，尤其是在大数据处理背景下，是 **Hadoop 分布式文件系统 (HDFS)**。它是 Apache Hadoop 生态系统的核心组成部分，专门设计用于在商品硬件（标准、廉价的计算机）集群上可靠地存储超大型文件。

HDFS采用主/从架构：

1. **NameNode（主节点）：** 可以把NameNode看作是整个文件系统的目录或索引。它本身不存储实际的文件数据，但它记录元数据：哪些文件存在，它们如何被分割成称为 **块** 的小片段，以及哪些DataNode存储这些块。它管理文件系统命名空间并控制对文件的访问。因为它保存了所有元数据，所以NameNode是集群中一个非常重要的部分。
2. **DataNode（从节点）：** 这些是实际存储数据块的机器。当您将一个大文件保存到HDFS时，它会被分解成固定大小的块（默认通常为128MB或256MB）。NameNode指示DataNode存储这些块的副本。例如，一个块可能会被复制到三个不同的DataNode上。

这种复制是HDFS容错性的基本。如果一个DataNode脱机（例如由于硬件故障），NameNode知道其块的其他副本位于何处，数据访问可以不中断地继续。

> HDFS架构的简化图。NameNode管理元数据，而DataNode在集群中存储复制的数据块（如块A和块B），以实现容错。

HDFS针对\*\*“一次写入，多次读取”\*\*的访问模式进行了优化。这意味着它非常适合存储只写入一次（如日志文件或传感器读数）然后多次读取进行分析的大型数据集。它通常不太适合需要对现有文件进行频繁、低延迟更新的场景，关系型数据库或NoSQL数据库能更好地处理这类情况。

### 在数据工程中的作用

像HDFS这样的分布式文件系统作为许多大数据操作的基础层：

- **大规模存储：** 它们提供存储海量数据的能力，这些数据量超出了单机的限制。这包括从各种来源收集的原始数据、处理步骤中产生的中间数据以及可供分析的最终处理数据集。
- **数据处理集成：** 专为大规模数据处理设计的框架，例如Apache Spark或MapReduce，旨在高效地从像HDFS这样的分布式文件系统读取数据。它们可以直接在DataNode存储数据的地方并行处理数据，最大限度地减少网络上的数据移动。
- **数据湖基础：** HDFS通常构成数据湖的存储骨干，以其原始格式保存各种数据类型（结构化、半结构化、非结构化），然后它们可能被结构化或加载到数据仓库中。

尽管功能强大，但HDFS与我们之前讨论的数据库有所不同。它在文件层面操作，而非记录层面。查询文件中的特定记录通常需要处理框架来读取和解析文件。它也与对象存储（我们将在接下来介绍）不同，对象存储是一种在云环境中常被选择的技术，因其可扩展性、持久性和API可访问性，有时会取代或补充HDFS。

### 注意事项

在使用或考虑像HDFS这样的分布式文件系统时，请记住以下几点：

- **延迟：** 相比本地文件系统，访问数据可能具有更高的延迟，尤其是对于小文件，这主要是由于网络通信和分布式特性。
- **复杂性：** 管理HDFS集群需要操作开销（设置、监控、维护），尽管云中的托管HDFS服务可以缓解这一点。
- **用例匹配度：** 它擅长存储和访问用于批处理和分析的超大型文件。它不适合低延迟的随机读写或管理数百万个微小文件。

总之，像HDFS这样的分布式文件系统是数据工程工具包的重要组成部分，提供可扩展且具有弹性的存储，以处理现代数据分析和AI应用中常见的海量数据集。它们是大规模数据处理通常发生的基础。

## 参考资料

- [HDFS Architecture Guide](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html) — The Apache Software Foundation (2025)
  Publisher: The Apache Software Foundation
  解释Hadoop分布式文件系统基本设计原则和架构的官方文档。
- [The Google File System](http://doi.acm.org/10.1145/945445.945450) — Sanjay Ghemawat, Howard Gobioff, and Shun-Tak Leung (2003)
  Journal: SOSP '03: Proceedings of the nineteenth ACM symposium on Operating systems principles; Publisher: ACM; Pages: 29-43; DOI: [10.1145/945445.945450](https://doi.org/10.1145/945445.945450)
  描述Google分布式文件系统的开创性论文，该系统对HDFS的设计和开发产生了巨大影响。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  这本著作涵盖了数据系统的基础概念，其中包括HDFS等分布式文件系统，并将其置于构建可伸缩和弹性应用的更广阔范围中。
