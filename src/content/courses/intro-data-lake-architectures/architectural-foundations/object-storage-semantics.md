---
course: "intro-data-lake-architectures"
chapter: "architectural-foundations"
lesson: "object-storage-semantics"
sourceId: 7996
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-1-architectural-foundations/object-storage-semantics"
title: "对象存储语义"
description: "对对象存储中的最终一致性、可变性和层次结构与传统文件系统进行技术比较。"
order: 2
plots: ["plots/7996-0.json"]
sourceHash: "01be287f918f7f7cbc6ad78cf2aecb80b88e168a4784d47624287770df16ab94"
sourceCorrections: []
---

现代数据架构大量使用亚马逊 S3、Azure Blob 存储和谷歌云存储等云存储服务。虽然这些服务常提供类似于标准文件管理器（带文件夹和文件）的图形用户界面，但其底层技术与本地硬盘或 Linux 服务器上的 POSIX 兼容文件系统截然不同。对这些差异的误解是导致数据湖部署中性能下降的主要原因。

### 键值存储的扁平结构

文件系统与对象存储之间最主要的区别在于没有真实的目录层级结构。在传统文件系统中，目录是一个独立的实体，包含指向文件或子目录的指针。而在对象存储中，结构是扁平的。

数据以对象形式存储在存储桶（或容器）中。每个对象由一个唯一标识符识别，通常称为键。当您将文件保存到 `s3://my-bucket/data/sales/file.parquet` 时，服务不会创建名为 `data` 的文件夹，然后是名为 `sales` 的子文件夹，最后将文件放入其中。相反，它只是创建一个具有键 `data/sales/file.parquet` 的单个对象。

您在管理控制台中看到的“文件夹”是虚拟的。界面会解析键中的斜杠 `/`，以方便人们使用的方式呈现分层视图。这种架构差异对数据工程操作有直接影响，尤其是在管理分区结构时。

### 重命名操作的代价

扁平命名空间的含义在文件管理操作中变得显而易见。在 POSIX 文件系统中，移动或重命名目录是复杂度为 $O(1)$ 的原子操作。操作系统仅更新目录指针到新位置。目录内数据的大小不影响完成操作所需的时间。

在对象存储中，因为目录实际不存在，您无法“重命名”目录。要模拟将文件夹从 `temp/` 移动到 `final/`，系统必须对每个共享该前缀的单个对象执行复制-删除工作流。

如果您在临时位置有 10,000 个文件并希望将它们升级到生产表，对象存储必须：

1. 列出所有带有源前缀的对象。
2. 将每个对象单独复制到新的键目标。
3. 删除原始对象。

时间复杂度从常数时间变为线性时间，这与文件数量 ($N$) 和数据大小 ($S$) 有关。

$Cost_{move} \approx N \times (Latenc\gamma_{request}) + S \times (Throughput_{network})$

此操作既非原子性也非快速。如果过程进行到一半失败，数据会分散在源和目标之间，导致数据损坏或不一致状态。这就是为什么 Delta Lake 和 Apache Iceberg 等现代表格式完全避免目录重命名，转而依赖元数据层来跟踪哪些文件属于表的当前状态。

> 文件系统中元数据指针更新与对象存储中所需的复制-删除机制的对比。

### 不可变性与追加操作

云存储中的对象是不可变的。对象一旦写入，就不能修改。您无法在 S3 中打开文件，定位到特定字节偏移量并覆盖值，也无法向现有对象末尾追加数据。

这一限制决定了数据管道如何摄取信息。在传统环境中，日志应用程序可能会持续向单个 `server.log` 文件追加行。在数据湖环境中，尝试追加数据将需要读取整个现有对象，在内存中添加新行，并将整个对象重新写入存储。

为了高效处理数据摄取，数据工程师使用特定模式：

- **微批处理 (Micro-batching)：** 在内存中缓冲传入记录，并每隔几分钟将它们作为新的不可变文件（例如，`part-001.parquet`）写入。
- **日志结构合并 (Log-Structured Merge)：** 将存储视为一系列不可变事件，而非可变表。

### 一致性模型

分布式系统必须平衡可用性和一致性。历史上，对象存储提供“最终一致性”。这意味着如果您写入文件并立即尝试列出它，系统可能会报告该文件尚未存在。写入操作必须跨云提供商数据中心内的多个物理副本传播。

如今，AWS S3、谷歌云存储和 Azure Blob 存储等主要提供商为新对象创建和覆盖提供强一致性。当您在 `PUT` 请求后收到成功响应 (HTTP 200) 时，随后的 `GET` 或 `LIST` 请求保证能看到该对象。

然而，需要明确的是，强一致性通常适用于对象本身。位于存储之上的元数据目录或搜索索引仍可能遇到传播延迟。如果您依赖外部 Hive Metastore 来跟踪分区，存储中可能已存在文件，但查询引擎在 Metastore 更新之前不会知道它。

### 网络延迟与吞吐量 (throughput)

与对象存储的交互通过 HTTP/HTTPS API 进行。每个读写操作都涉及 DNS 解析、TCP 握手和 SSL 协商。与本地磁盘 I/O 相比，这导致每个操作的延迟显著更高。

本地固态硬盘可能提供亚毫秒级的访问时间，但对对象存储的请求通常会产生 20 到 100 毫秒的首字节延迟。这种延迟使得对象存储不适合需要随机读取数千个小文件的工作负载。

相反，对象存储在吞吐量方面表现出色。因为存储后端分布在庞大集群中，您可以通过并行请求实现每秒数 TB 的聚合带宽。分析引擎通过读取更少、更大的文件（通常 100MB 到 1GB）而非许多小文件来对此进行优化。



![吞吐量与延迟的权衡](plots/7996-0.json)



> 对象存储与本地固态硬盘在运行特性上的对比，前者具有高延迟但大规模并行吞吐量。

### 分段上传

为了高效处理大型数据集，对象存储 API 实现了分段上传。此功能允许将单个大型对象（S3 上最大 5TB）作为一组不同的部分上传。这些部分可以并行上传，以最大化网络带宽。

如果单个部分的上传失败，客户端只需重试该特定部分，而不是重新开始整个 5TB 传输。一旦所有部分上传完成，存储服务会将它们逻辑地合并成一个对象。大多数高级 SDK 和 Spark 等工具会自动处理此过程，但理解其工作原理很有必要，尤其是在调试失败的作业或调整内存管理的缓冲区大小时。

通过遵守这些语义、不可变性、扁平命名空间和高延迟 HTTP 接口，工程师可以设计数据摄取和查询层，使其顺应存储系统特性而非逆反其特性。

## 参考资料

- [How Amazon S3 Works](https://docs.aws.amazon.com/AmazonS3/latest/userguide/how-s3-works.html) — Amazon Web Services (AWS) (Ongoing)
  Publisher: AWS
  阐述了Amazon S3的基本原理，包括其对象模型、扁平命名空间、不变性、一致性保证和操作特点。
- [Blob storage concepts](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-introduction) — Microsoft Azure Documentation (2023)
  Publisher: Microsoft
  详细介绍了Azure Blob存储的核心概念，包括容器、blob、存储结构和数据操作处理，与S3文档互补。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449370275/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了分布式系统基础的全面视角，包括存储引擎设计、一致性模型以及对理解对象存储至关重要的权衡。
