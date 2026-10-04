# 分布式文件系统 (HDFS, S3)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-8-building-managing-large-scale-datasets/distributed-file-systems)

[返回章节目录](README.md) · [返回课程目录](../README.md)

处理以太字节甚至拍字节衡量的数据集时，将它们存储在单台机器的文件系统上会变得不切实际，甚至不可能。庞大的数据量超出了本地磁盘容量，而在分布式训练期间，多个机器（工作节点）需要并发读取这些数据，这会造成一个很大的瓶颈。分布式文件系统（DFS）和云对象存储提供了处理这些挑战所需的基础设施，它们具有可扩展性、高可靠性和并行访问能力。

这些系统与本地文件系统根本不同，它们将数据分布在多个物理存储节点上，同时向用户或应用程序呈现统一的视图。这种分布实现了横向扩展；随着数据增长，你可以增加更多存储节点，而不会受到单台机器容量的限制。

### LLM分布式存储的核心要点

两个主要特点使分布式存储适合大规模数据管理：

1. **可扩展性和性能：** 分布式系统可以存储比任何单个节点多得多的数据。它们通常设计为高吞吐量 (throughput)，允许许多客户端（例如你的训练工作节点）并行读取数据块。这对于在训练期间持续为昂贵的GPU提供数据供给至关重要。特别是云对象存储，按需提供几乎无限的可扩展性。
2. **可靠性和容错性：** 存储拍字节的数据会增加遇到硬件故障（磁盘故障、节点中断）的可能性。分布式系统通过在多个节点上复制数据块（HDFS常见）或使用纠删码技术（对象存储常见）来降低这种风险。如果一个节点发生故障，数据仍然可以从其他副本重建或取回，从而数据持久性并防止因存储故障导致的训练中断。

### Hadoop分布式文件系统 (HDFS)

HDFS起源于Apache Hadoop生态系统，专为存储具有流式数据访问模式的超大文件而设计。它采用主/从架构：

- **NameNode（名称节点）：** 管理文件系统命名空间（元数据，如目录结构、文件权限和数据块位置）。它是中心协调器。
- **DataNodes（数据节点）：** 在本地磁盘上存储实际数据块，并向NameNode报告。

HDFS通常会将每个数据块（例如，128MB或256MB的块）在不同的DataNodes上复制三份以实现容错。

**优点：**

- 对大型顺序读取具有出色的吞吐量 (throughput)，非常适合处理大型数据文件。
- 成熟的生态系统集成，特别是与Apache Spark进行数据预处理。
- 可以使用商用硬件在本地部署。

**缺点：**

- NameNode可能成为性能瓶颈或单点故障（尽管存在高可用性配置）。
- 对随机访问或小文件工作负载的优化较少。
- 需要主动的集群管理和维护。

当LLM数据预处理流程严重依赖现有Hadoop/Spark基础设施，或者组织管理大型本地集群时，HDFS常被使用。

### 云对象存储 (AWS S3, GCS, Azure Blob Storage)

像Amazon S3（简单存储服务）、Google Cloud Storage（GCS）和Azure Blob Storage这样的云对象存储服务已经成为在云中存储大型数据集的事实标准。它们与传统文件系统的运作方式不同：

- **扁平命名空间：** 数据作为对象存储在容器中（S3/GCS中称为“桶”，Azure Blob Storage中称为“容器”）。对象通过唯一键识别（通常类似文件路径，例如`my-data/processed/part-0001.arrow`）。
- **API驱动：** 主要通过HTTP API（GET, PUT, LIST, DELETE）进行交互。
- **高持久性和可用性：** 这些是托管服务，旨在提供极高的持久性（例如，99.999999999%或“十一个九”）和可用性，将底层硬件管理和复制抽象化。

> 多个计算工作节点从中心分布式存储系统并发访问数据。

**优点：**

- 大规模可扩展性和弹性（按使用量付费）。
- 由云服务提供商管理的高持久性和可用性。
- 与自管理HDFS相比，运维负担更小。
- 与基于云的计算实例和训练平台直接集成。
- 支持数据分层以优化成本（例如，将旧数据移动到更便宜、不常访问的层级）。

**缺点：**

- 性能可能受计算与存储之间网络延迟的影响。
- 最终一致性模型（尽管强一致性越来越普遍）在某些读写后场景中可能需要仔细处理。
- 如果将大量数据移出云，数据传输成本（出站流量）可能是一个需要考虑的因素。

对于大多数基于云的LLM开发，对象存储是首选，因为它的可扩展性、易用性以及与更广泛的云生态系统的集成。

### 训练期间访问分布式存储

训练框架需要有效的方法从这些分布式系统中读取数据。在训练前简单地将数太字节的数据下载到每个工作节点的本地磁盘是不切实际的。相反，数据通常在训练期间直接从分布式存储进行流式传输。

像`fsspec`（文件系统规范）这样的库为各种存储后端提供了统一的Python接口，包括本地文件、HDFS、S3、GCS和Azure Blob Storage。PyTorch的`DataLoader`可以与利用这些库的自定义`Dataset`实现结合使用。

这里有一个使用`torch.utils.data.IterableDataset`从存储在S3类桶中的多个文本文件逐行流式传输数据的例子：

```python
import torch
import s3fs # 或者boto3，或其他相关库
import gzip
from torch.utils.data import IterableDataset, DataLoader

class S3StreamingTextDataset(IterableDataset):
    def __init__(self, bucket_name, prefix, shuffle_files=True):
        super().__init__()
        self.fs = s3fs.S3FileSystem() # 假设凭据已配置
        self.bucket = bucket_name
        self.file_paths = self.fs.glob(f"{bucket_name}/{prefix}/*.txt.gz")
        self.shuffle_files = shuffle_files
        print(f"在"
              f"s3://{bucket_name}/{prefix}" f"中找到了 {len(self.file_paths)} 个文件")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.file_paths

        if worker_info is not None:
            # 将文件分配给不同的工作节点
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_to_process = [
                f for i, f in enumerate(self.file_paths)
                if i % num_workers == worker_id
            ]

        if self.shuffle_files:
             # 为这个工作节点轮次打乱文件
             # 注意：要实现真正的打乱，请首先考虑打乱全局索引
             import random
             random.shuffle(files_to_process)

        for file_path in files_to_process:
            print(f"工作节点 {worker_info.id if worker_info else 0}: "
                  f"正在处理 {file_path}")
            try:
                # 使用s3fs直接打开文件
                with self.fs.open(file_path, 'rb') as f_remote:
                    with gzip.open(f_remote, 'rt', encoding='utf-8') as f_gz:
                        for line in f_gz:
                            # 如果需要，在这里预处理/分词行
                            # 为简单起见，只返回原始行
                            yield line.strip()
            except Exception as e:
                print(f"工作节点 {worker_info.id if worker_info else 0}: "
                      f"处理 {file_path} 时出错: {e}")
                # 决定如何处理错误：跳过文件、记录日志、还是抛出异常？
                continue

# 使用示例
# dataset = S3StreamingTextDataset(
#     bucket_name='my-llm-data',
#     prefix='processed_data'
# )
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     num_workers=4
# ) # 4 worker processes

# for batch in dataloader:
#     # 处理文本行批次
#     # print(batch)
#     pass
```

这个例子展示了流式传输：每个工作进程（`DataLoader`中的`num_workers`）被分配S3中文件的一个子集，并直接从对象存储逐行读取，同时进行解压。这避免了将整个数据集下载到本地，并允许在多个工作节点和文件之间进行并行处理。更完善的实现可能会以更大的块读取数据，使用像Apache Arrow或Parquet（在上一节中讨论过）这样的优化文件格式，并实施更强大的混洗策略。

选择合适的分布式存储系统是重要的第一步。对于现在开始的大多数项目，特别是那些利用云计算的项目，S3、GCS或Azure Blob等托管对象存储提供了可扩展性、持久性和易用性的吸引人组合，并且与专为流式访问设计的现代分布式训练框架和数据加载器很好地集成。

## 参考资料

- [The Google File System](https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf) — Sanjay Ghemawat, Howard Gobioff, Shun-Tak Leung (2003)
  Journal: Proceedings of the nineteenth ACM symposium on Operating systems principles; Pages: 29-43; DOI: [10.1145/945445.945451](https://doi.org/10.1145/945445.945451)
  一篇基础性论文，介绍了分布式文件系统的原则，包括可扩展性、容错性和大型顺序数据访问的设计选择，对HDFS产生了影响。
- [Apache Hadoop HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html) — The Apache Software Foundation (2025)
  官方文档，描述了Hadoop分布式文件系统的架构、设计和操作方面。
- [Amazon Simple Storage Service (S3) Documentation](https://docs.aws.amazon.com/s3/index.html) — Amazon Web Services (AWS) (2024)
  官方文档，全面介绍了亚马逊S3及其功能、API、数据模型和云对象存储的最佳实践。
- [fsspec: Filesystem interfaces for Python](https://filesystem-spec.readthedocs.io/en/latest/) — Martin Durant and other contributors (2025)
  fsspec的官方文档，这是一个Python库，为访问各种文件系统（包括本地、HDFS和S3等云对象存储）提供统一接口。

---

[上一节](01-%E6%95%B0%E6%8D%AE%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F%EF%BC%88%E6%96%87%E6%9C%AC%E3%80%81Arrow%E3%80%81Parquet%EF%BC%89.md) · [下一节](03-%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95%E7%94%A8%E4%BA%8E%E9%AB%98%E6%95%88%E6%A3%80%E7%B4%A2.md)
