---
course: "planning-optimizing-ai-infrastructure"
chapter: "designing-on-premise-ai-infrastructure"
lesson: "gpu-interconnect-technologies"
sourceId: 6981
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure/gpu-interconnect-technologies"
title: "GPU 互联技术"
description: "比较 NVLink 和 PCIe 等 GPU 互联技术及其对多 GPU 训练性能的影响。"
order: 3
plots: ["plots/6981-0.json"]
sourceHash: "f8fccd40ecf584e416f0864e05e5a84ceef282aff9e5c0b4ea2e166b59d38028"
sourceCorrections: []
---

在服务器中连接多颗 GPU 时，它们高效通信的能力变得和单个的算力 (compute)一样重要。这种通信通道，或者说互联，是数据在 GPU 之间以及在 GPU 和 CPU 之间传输的物理通路。训练作业的总耗时是计算时间与通信时间之和：$T_{total} = T_{compute} + T_{communication}$。慢速互联会造成严重的瓶颈，导致您昂贵的 GPU 等待数据，并大幅增加 $T_{communication}$。构建本地系统时会遇到的两种主要互联技术是：标准 PCI Express 总线和英伟达的专用 NVLink。

### 标准：PCI Express (PCIe)

外围组件互联高速接口，即 `PCIe`，是一种标准的高速接口，用于在主板上连接处理器与 GPU、高速网卡和 NVMe 硬盘等外围设备。每台现代服务器和台式计算机都使用它。当您将 GPU 安装到主板上的插槽中时，就是将其连接到系统的 `PCIe` 总线。

对于单 GPU 配置，`PCIe` 完全够用。GPU 与 CPU 通信以获取数据和指令，执行其并行计算，并将结果传回。瓶颈（如果有的话）通常出现在其他地方。

挑战出现在多 GPU 配置中。在典型的服务器主板中，通过 `PCIe` 总线通信的 GPU 必须通过 CPU 路由其流量。如果 GPU 1 需要向 GPU 2 发送更新，数据路径通常如下所示：GPU 1 -> PCIe 总线 -> CPU -> PCIe 总线 -> GPU 2。这种间接路径引入了显著的延迟，并消耗了连接到 CPU 的有限 `PCIe` 带宽。尽管像 `PCIe` 对等传输 (P2P) 这样的技术可以实现更直接的传输，但性能可能不稳定，并且严重依赖于主板的拓扑结构。

`PCIe` 连接的带宽由其版本（例如 3.0、4.0、5.0）和使用的通道数量（例如 x8、x16）决定。例如，`PCIe 4.0` x16 插槽提供 64 GB/s 的理论双向带宽，而 `PCIe 5.0` 则将其翻倍到 128 GB/s。尽管令人印象深刻，但此带宽在所有连接到 CPU 的设备之间共享，并且在密集的多 GPU 同步期间可能达到饱和。

### 高速替代方案：英伟达 NVLink

为解决 `PCIe` 在多 GPU 工作负载方面的局限性，英伟达开发了 `NVLink`，这是一种专有的高速互联，提供 GPU 之间的直接通信链路。支持 `NVLink` 的 GPU 不再通过 CPU 路由流量，而是可以直接交换数据，就像在专用高速公路上一样。这大幅降低了延迟，并提供了专门用于 GPU 间通信的巨大带宽提升。

下图说明了标准 `PCIe` 系统与 `NVLink` 桥接系统之间数据路径的差异。

> 在仅 `PCIe` 配置（左侧）中，GPU 之间的通信由 CPU 协调。而通过 `NVLink` 桥接（右侧），GPU 获得直接、高带宽连接，从而释放 `PCIe` 总线用于与 CPU 和存储器通信。

这种直接路径改变了模型并行等分布式训练技术的局面，在这些技术中，大型模型的不同层分布在不同的 GPU 上，并且必须持续交换中间激活值。它对数据并行也非常有效，在数据并行中，每一步之后梯度必须在所有 GPU 之间平均。通过最小化 $T_{communication}$，`NVLink` 使训练能够更有效地跨多个 GPU 扩展。

### 带宽对比：定量分析

这两种技术之间的性能差异非常明显。`NVLink` 带宽随着每一代而增加，持续领先于当时可用的相应 `PCIe` 标准。



![GPU 互联双向带宽对比](plots/6981-0.json)



> 常见 `PCIe` 和 `NVLink` 代际的理论双向带宽对比。请注意，`NVLink` 值（例如第四代 900 GB/s）通常适用于 HGX 系统等高端服务器节点中一组全连接的 GPU，其中每颗 GPU 使用多个链路。

该图表显示了数量级的差异。第四代 `NVLink`（与 H100 GPU 配合使用）的 900 GB/s 带宽是 `PCIe 5.0` 的 7 倍以上。这种海量带宽使得在合理的时间内训练当前最大的模型成为可能。

### 系统设计中的实际考量

互联技术的选择直接影响您的服务器构建、预算和性能。

- **工作负载依赖性：** 如果您的主要工作负载涉及单 GPU 任务或通信非常不频繁的分布式任务，那么 `NVLink` 的额外开销可能不划算。具有多个 `PCIe 4.0` 或 `5.0` 插槽的系统可以成为并行运行许多独立实验的经济有效解决方案。
- **扩展需求：** 如果您的目标是加速跨多个 GPU 训练单个大型模型，那么 `NVLink` 实际上是必需的。降低 $T_{communication}$ 带来的性能提升将通过缩短训练周期和提高 GPU 利用率，远远超过初始硬件成本。
- **硬件兼容性：** `NVLink` 是高端数据中心级英伟达 GPU（如 A100 和 H100 系列）的特点。它不适用于大多数消费级 GeForce 显卡。此外，要使用 `NVLink` 桥接连接 GPU，您需要主板其 `PCIe` 插槽物理间距正确，以适应刚性桥接器。对于两颗以上 GPU 的系统，您通常需要专门的服务器平台，例如英伟达认证系统，专门为高密度 `NVLink` 连接的 GPU 配置而设计。

最终，选择基于 `PCIe` 的设计还是基于 `NVLink` 的设计是一个权衡问题。您必须权衡前期资本支出与最严苛 AI 工作负载的性能需求。对于认真进行大规模模型开发的组织来说，投资 `NVLink` 基础设施就是投资于更快的迭代和更具能力的研发平台。

## 参考资料

- [NVIDIA H100 GPU Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — NVIDIA Corporation (2022)
  Publisher: NVIDIA
  介绍了H100 GPU的架构，包括第四代NVLink的详细信息及其在加速多GPU通信中的作用。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Elsevier; Pages: Chapter 6: The Warehouse-Scale Computer
  一本经典的计算机体系结构教材，提供了关于I/O子系统和PCIe等互连技术的理解。
- [A Survey of Communication Bottlenecks in Distributed Deep Learning](https://dl.acm.org/doi/10.1145/3418544) — Youngeun Kang, Jianzong Li, Kai Zeng, Dingwen Zeng (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 53; Pages: Article 121; DOI: [10.1145/3418544](https://doi.org/10.1145/3418544)
  此综述识别并分析了分布式深度学习中的各种通信瓶颈，包括GPU互连（如PCIe和NVLink）的作用。
- [Deep Learning Systems: Algorithms, Architectures, and Optimizations for Big Data](https://www.routledge.com/Deep-Learning-Systems-Algorithms-Architectures-and-Optimizations-for-Big-Data/Krishna/p/book/9780367468641) — K. M. Krishna (2021)
  Publisher: CRC Press; Pages: Chapter 3: Deep Learning Architectures and System Design
  为深度学习系统提供了架构要求方面的背景，包括高速互连在多GPU性能扩展中的作用。
