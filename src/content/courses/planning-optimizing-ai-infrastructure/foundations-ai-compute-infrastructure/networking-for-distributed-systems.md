---
course: "planning-optimizing-ai-infrastructure"
chapter: "foundations-ai-compute-infrastructure"
lesson: "networking-for-distributed-systems"
sourceId: 6974
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/networking-for-distributed-systems"
title: "分布式系统的网络考量"
description: "理解构建高效分布式AI系统所需的网络需求，包括带宽和延迟。"
order: 8
plots: []
sourceHash: "a37b68db98d86357dea21ee4dd1f511ad1d7a5830d06c0f1f7c3b622a1e6369f"
sourceCorrections: []
---

当在单台机器上扩展AI工作负载时，网络性能对系统效能的影响与处理器相当。一组强大的GPU如果不能有效通信，其性能会大大降低。在分布式系统中，多台机器合作解决一个问题时，网络是将它们连接起来的纽带。它负责将训练数据传输到节点、同步模型参数 (parameter)以及收集结果。如果这个纽带缓慢或不可靠，你昂贵的计算资源大部分时间都将用于等待。

本节研究网络性能的两个主要衡量标准：带宽和延迟，并说明了它们对常见AI训练模式的重要性。

### 带宽和延迟：网络性能的两大要素

在评估网络时，我们关注两个基本特点：

1. **带宽：** 通常被称为吞吐量 (throughput)，带宽是指在给定时间内，网络连接可以传输的最大数据量。通常以比特每秒（如吉比特每秒 Gbps 或太比特每秒 Tbps）为单位测量。可以把带宽想象成高速公路上的车道数量。一条10车道的高速公路（高带宽）可以比2车道公路（低带宽）同时承载更多的车辆（数据）。对于AI而言，高带宽对于在节点间传输大型对象，例如数据集、模型检查点或全部模型参数 (parameter)必不可少。
2. **延迟：** 延迟是指单个数据包从源头传输到目的地所需的时间。它是衡量延迟的指标，通常以毫秒（ms）或微秒（µs）表示。在我们的高速公路类比中，延迟是指一辆车从起点到终点所需的时间，无论有多少车道。低延迟对于涉及频繁、小型、往复通信的操作非常重要。

在分布式训练中，特别是数据并行场景下，节点在每个训练步骤后必须不断同步模型更新（梯度）。这涉及大量小型、频繁的消息。如果延迟很高，每个同步步骤都会引入显著延迟。GPU完成工作后会等待网络，导致利用率低下。

> 一个理想的AI网络结合了高带宽（宽管道）以传输大量数据，并结合低延迟（短管道）以确保同步任务的快速通信。

### 分布式训练中的通信模式

网络的效率取决于它对应用程序特定通信模式的处理能力。在分布式深度学习 (deep learning)中，最具挑战性的模式之一是 `All-to-All`。在梯度同步期间，每个GPU节点需要将其计算出的梯度发送给所有其他节点。

想象一个由8个GPU组成的集群。在一个训练步骤之后，每个GPU都拥有总梯度的一部分。为了准备下一个步骤，每个GPU都需要完整的平均梯度。这需要复杂的数据混洗，其中每个节点同时与其他所有节点通信。

`All-to-All` 交换中的高延迟会造成复合延迟，因为最慢的连接会拖累整个集群。这就是为什么大型AI系统中会使用专用网络硬件和库。

### 高性能网络技术

虽然标准的企业以太网（1 GbE 或 10 GbE）足以应对一般IT任务，但对于严肃的分布式AI工作负载来说通常不够。高性能计算（HPC）环境长期以来一直依赖更先进的互连技术。

- **InfiniBand：** 这是一种为极低延迟和高带宽而设计的高性能网络标准。它是构建大型AI超级计算机的常见选择。InfiniBand的一个重要特性是它对\*\*远程直接内存访问（RDMA）\*\*的支持。
- **RDMA（远程直接内存访问）：** 在传统的网络堆栈中，将数据从一台机器移动到另一台机器需要发送端和接收端CPU及操作系统的多个步骤参与。这个过程会引入显著延迟并消耗CPU周期。RDMA允许一台机器的网络接口卡（NIC）直接将数据写入另一台机器的内存（RAM甚至VRAM），从而绕过CPU和操作系统。这显著降低了延迟，并使CPU可以执行其他工作。
- **融合以太网上的RDMA (RoCE)：** 这是一种允许您在标准以太网网络上实现类似RDMA性能的协议，如果底层网络配置得当以实现“无损”，它能提供与InfiniBand竞争的替代方案。

了解这些网络基础知识是第一步。正如您将在后续关于设计本地和云基础设施的章节中看到，网络技术的选择对系统性能和总成本都有直接而显著的影响。投资于低延迟、高带宽的网络可以通过确保您昂贵的GPU资源始终忙于计算而非等待来收回成本。

## 参考资料

- [InfiniBand Overview: Scalable Interconnect for Data Center and Cloud](https://docs.nvidia.com/networking/display/INFBNDDOC/InfiniBand+Overview%3A+Scalable+Interconnect+for+Data+Center+and+Cloud) — NVIDIA (2023)
  Publisher: NVIDIA Docs Hub NVIDIA Networking
  NVIDIA的技术简介，解释了InfiniBand和RDMA技术，强调了它们对高性能计算和AI工作负载的优势。
