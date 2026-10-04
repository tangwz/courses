---
course: "planning-optimizing-ai-infrastructure"
chapter: "foundations-ai-compute-infrastructure"
lesson: "storage-solutions-for-ai-datasets"
sourceId: 6973
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/storage-solutions-for-ai-datasets"
title: "AI数据集存储方案"
description: "比较各种存储方案，从本地SSD到分布式文件系统，以管理大型AI数据集。"
order: 7
plots: ["plots/6973-0.json"]
sourceHash: "b55623049f4be2a74b1b7de10e87301e89ff6b027bb5326f34fcef24b792dccb"
sourceCorrections: []
---

高性能GPU能够以惊人的速度处理数据，但其效率完全取决于数据传输的速度。当存储系统无法跟上计算引擎的需求时，就会产生输入/输出（I/O）瓶颈，导致昂贵的加速器闲置。对于机器学习 (machine learning)而言，数据集可达千兆字节（GB）至拍字节（PB），选择合适的存储方案是基础设施设计中的重要组成部分。

AI的理想存储系统需平衡三个性能指标：吞吐量 (throughput)、延迟和容量。

- **吞吐量** 是数据传输速率，通常以兆字节每秒（MB/s）或千兆字节每秒（GB/s）衡量。高吞吐量对于快速载入图像或视频帧等大数据批次到内存中非常重要。
- **延迟** 是获取第一份数据所需的时间延迟，通常以微秒（$\mu$s）或毫秒（ms）衡量。在训练由数百万个小文件组成的数据集时，低延迟非常重要，因为查找每个文件所花费的时间会累积起来。
- **容量** 是系统可存储的数据总量。这是一个由数据集大小决定的直接要求。

存储方案大致分为本地存储、网络附加存储和对象存储，每种都提供了这些性能指标的不同组合。

### 本地存储：高速直接访问

本地存储直接连接到执行计算的机器。这种物理接近性提供了尽可能低的延迟，并且凭借适当的技术，可实现最高的吞吐量 (throughput)。

- **硬盘驱动器（HDDs）：** 这些是传统的旋转磁碟。虽然它们每千兆字节成本最低且容量大，但其机械特性导致高延迟和慢速传输。HDD通常过慢，不适合作为活跃训练数据集的主要存储，更适合长期归档或冷存储。
- **SATA固态硬盘（SSDs）：** SSD使用闪存且无活动部件，因此其延迟大幅降低，吞吐量远高于HDD。SATA SSD是工作站处理中小型数据集（可存放在单个驱动器上）的良好选择。
- **NVMe固态硬盘（SSDs）：** 非易失性内存高速（NVMe）是一种专门为SSD设计的协议，用于通过高速PCIe总线直接与CPU通信。这绕过了旧的、较慢的SATA接口，实现了惊人的速度。现代NVMe驱动器可提供超过7,000 MB/s的顺序读取速度，使其成为单节点训练的首选，在单节点训练中需要最大I/O性能来满足GPU的吞吐需求。



![存储性能对比](plots/6973-0.json)



> 常见本地存储类型的性能特征。请注意延迟使用对数刻度，这突显了设备类型之间数量级的差异。

### 网络与分布式存储

当数据集过大，无法存储在单台机器上，或需要多个计算节点同时访问以进行分布式训练时，您必须转向基于网络的解决方案。

- **网络附加存储（NAS）：** NAS是一种专用文件存储服务器，它通过局域网（LAN）向其他机器提供存储，通常使用NFS（网络文件系统）等协议。配备高速网络（10GbE或更快）的高性能NAS可以有效地向小型机器集群提供数据。然而，如果许多客户端同时发出请求，NAS设备本身可能成为单点故障和性能瓶颈。
- **分布式文件系统：** 对于大规模操作，分布式文件系统通常是解决方案。这些系统将来自多个服务器（节点）的存储汇集到一个统一的命名空间中。数据分散在各个节点上，一个文件可以同时从多个磁盘并行读取，从而提供极高的聚合吞吐量 (throughput)。Ceph或Lustre等系统是高性能计算（HPC）和大型AI集群中使用的实例。它们设置和管理复杂，但提供单个NAS所不具备的扩展性和容错性。

### 云对象存储

云提供商提供一种高度可扩展且持久的存储方法，即对象存储。Amazon S3、Google Cloud Storage (GCS) 和 Azure Blob Storage 等服务是云中数据存储的核心部分。

对象存储不使用文件夹和文件的分层文件系统，而是将数据作为“对象”在扁平地址空间中管理。每个对象由数据本身、一些元数据和一个唯一ID组成。

- **优点：** 对象存储提供几乎无限的容量、高持久性（数据在多个位置复制），以及存储海量数据的相对较低的成本。这使其成为构建“数据湖”来存储所有原始和已处理数据集的理想之地。
- **缺点：** 通过互联网从对象存储访问数据会引入更高的延迟，与本地或基于局域网的存储相比。因此，它通常不用于高性能训练期间的直接访问。常见模式是使用对象存储作为持久的真实数据源，并在训练任务开始前，将所需的数据子集复制到计算实例上更快的本地NVMe存储中。

> AI存储方案的层级结构。性能通常会随着数据远离计算核心而下降，以速度换取更大的规模和共享能力。

存储方案的选择涉及权衡。对于在笔记本电脑上工作的数据科学家来说，一个大容量的内置NVMe驱动器可能就足够了。对于一家初创公司来说，构建其第一个专用AI服务器时，由NVMe驱动器组成的RAID阵列是一个强大的起点。对于运行在数十个GPU上进行分布式训练的大型企业来说，分布式文件系统或结合云对象存储与高性能本地缓存的混合方法变得必要。理解这些权衡对于设计能够处理您的数据而不仅仅是模型的IT架构是根本性的。

## 参考资料

- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  对机器学习系统设计提供了全面的视角，包含数据管理、存储考量以及生产级AI的架构选择。
- [Storage for AI: Accelerating Your Deep Learning Workloads](https://www.weka.io/resources/storage-for-ai-accelerating-your-deep-learning-workloads/storage-for-ai-accelerating-your-deep-learning-workloads.pdf) — WekaIO (2022)
  Publisher: WekaIO
  分析深度学习对存储的特定要求，讨论NVMe、并行文件系统的作用，以及优化GPU加速工作负载I/O性能的方法。
