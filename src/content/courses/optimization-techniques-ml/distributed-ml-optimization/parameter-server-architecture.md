---
course: "optimization-techniques-ml"
chapter: "distributed-ml-optimization"
lesson: "parameter-server-architecture"
sourceId: 1291
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-5-distributed-ml-optimization/parameter-server-architecture"
title: "参数服务器架构"
description: "协调分布式训练的参数服务器模型说明。"
order: 2
plots: []
sourceHash: "b4654f14cf395e281ab8eea4a25b36a7414da05bbdbc621a1f9b2f69f66be0c4"
sourceCorrections: []
---

训练大型机器学习 (machine learning)模型通常需要将计算分布到多台机器上。一种用于协调这种分布式工作的基础且被广泛采用的模式是**参数 (parameter)服务器 (PS)** 架构。它在模型参数的存储和更新与计算密集型梯度计算之间提供了逻辑上的分离。

设想您有一个大型模型，或许包含数十亿参数，大到无法在一台机器的内存中轻松容纳；或者数据集如此庞大，按顺序处理会花费过多的时间。参数服务器方法通过为计算集群中的不同节点分配特定角色来解决此问题。

### 架构组成部分

核心思想很简单：将模型状态与梯度计算分离。

1. **参数 (parameter)服务器：** 这些节点负责维护模型参数的全局状态。可以将它们视为一个分布式、可能分片的键值存储，其中键标识参数块（例如，神经网络 (neural network)的层），值是实际的参数张量 ($W$)。它们的主要任务是响应来自工作节点的参数请求，并聚合和应用从工作节点接收到的更新。根据模型大小，参数可能会在多个服务器节点之间进行分区（分片），以实现可伸缩性和弹性。
2. **工作节点：** 这些节点执行实际的训练工作。每个工作节点通常持有一个模型结构的副本（但不一定同时持有所有参数），并处理训练数据的一个子集（一个mini-batch）。工作节点的工作流程通常包括：
   - 从参数服务器拉取其计算所需的当前参数 ($W$)。
   - 根据其本地mini-batch数据和拉取的参数计算梯度 ($\nabla L(W)$)。
   - 将计算出的梯度（或参数更新）推回参数服务器。

### 工作流程

让我们看一下参数 (parameter)服务器设置中典型的数据流：

> 参数服务器架构：工作节点从参数服务器（可能分片）拉取最新参数 ($W$)，使用其本地数据批次计算梯度 ($\nabla L(W)$)，并将这些梯度推回服务器。服务器聚合梯度并更新参数。

参数服务器处理梯度的聚合（例如，简单的平均或应用更复杂的更新规则，如 Adam，这还需要存储优化器状态，如动量），并更新参数的主副本。这个循环重复许多次迭代。

### 优点和注意事项

参数 (parameter)服务器架构具有以下优点：

- **可扩展性：** 它能很好地扩展到非常大的模型，因为参数本身分布在服务器节点上。添加更多服务器可以增加参数容量。
- **灵活性：** 它自然支持异步训练（我们将在下一节讨论），其中工作节点无需相互等待，这可能导致更高的硬件利用率。
- **解耦：** 将参数存储与计算分离简化了工作节点的逻辑。

然而，它也带来了一些潜在问题：

- **通信瓶颈：** 参数服务器可能成为瓶颈，尤其是在许多工作节点同时尝试拉取参数或推送梯度时。工作节点和服务器之间的网络带宽通常是限制因素。
- **一致性：** 在异步设置中，工作节点可能会拉取与由其他工作节点推送的更新相比略微过时（"陈旧"）的参数。这需要在算法设计和分析中仔细考虑。
- **容错性：** 服务器可能成为单点故障，尽管复制和分片策略可以缓解这种情况。

了解参数服务器架构对于掌握许多分布式机器学习 (machine learning)系统非常重要。虽然存在替代方案和变体（例如后面会提到的使用 All-Reduce 的去中心化方法），但 PS 模型为大规模优化的工作划分提供了一个清晰的框架。我们将接下来讨论该框架内同步和异步更新之间的权衡。

## 参考资料

- [Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf) — Mu Li, Yuandong Tian, Anupam Singh, Arun Kumar, and James Dean (2014)
  Journal: Proceedings of the 11th USENIX Symposium on Operating Systems Design and Implementation (OSDI '14); Pages: 583-596
  介绍了用于分布式机器学习的基础参数服务器架构。
- [Deep Learning with Stochastic Gradient Descent: A Comparative Study of Downpour SGD, One-Shot Averaging, and Asynchronous SGD](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37731.pdf) — Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Maithra Raghu, Moayad Al-Ghamdi, Vineet Agrawal, Andrew Narayanan, and George Adorno (2012)
  Journal: NIPS Workshop on Optimization for Machine Learning
  讨论了Downpour SGD，一种早期的异步参数服务器实现，它解决了数据陈旧和通信问题。
