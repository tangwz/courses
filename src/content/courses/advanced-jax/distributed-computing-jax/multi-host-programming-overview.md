---
course: "advanced-jax"
chapter: "distributed-computing-jax"
lesson: "multi-host-programming-overview"
sourceId: 2859
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/multi-host-programming-overview"
title: "多主机编程简介"
description: "JAX 并行性如何从单机扩展到多台机器（如 TPU Pods）的介绍。"
order: 8
plots: []
sourceHash: "9e5f196effbb895f82019a14e565abfe9ae30ae7726ab9e42428c0e63b859370"
sourceCorrections: []
---

`pmap`在JAX中广泛应用于在连接到*一台*机器或主机上的多个加速器（GPU或TPU）之间进行并行计算。这种方法对许多任务非常有效，能够通过数据并行实现显著加速。然而，最严峻的机器学习 (machine learning)挑战，特别是训练拥有数十亿或数万亿参数 (parameter)的基础模型，需要的计算资源远远超出了单台主机所能提供的范围，即使是配备了多个加速器的主机也无法满足。这促使我们转向多主机编程。

JAX中的多主机编程将我们之前讨论过的并行性思路进行了扩展，以便通过网络协调多台独立机器上的计算。设想一下一个机器集群，每台机器都可能配备多个GPU或TPU。目标是调度这些机器，共同完成一个大型计算任务。

### 扩展到多个节点

虽然与单主机`pmap`类似，多主机执行带来了重要的实际考量：

1. **网络通信：** 在单个主机内部，设备通常通过PCIe或NVLink等快速互连方式进行通信。主机之间，通信则依赖于网络（例如以太网、InfiniBand或专用TPU互连）。网络延迟和带宽通常比主机内部互连差几个数量级，这使得跨主机通信可能成为性能瓶颈。高效的集合通信算法变得非常重要。
2. **同步与协调：** 你不再处理由单个操作系统管理的线程。相反，你有运行在不同机器上的独立Python进程，它们需要协调各自的执行。这包括确保进程正确启动、相互发现，并在特定时刻（例如在集合操作期间）进行同步。
3. **基础设施与作业启动：** 设置和管理多主机环境更为复杂。你需要有方法在所有参与主机上启动相同的JAX脚本、配置网络，并可能处理故障。这通常涉及使用集群管理系统（如SLURM）或云服务商专用工具（例如Google Cloud管理TPU Pod切片的工具）。JAX本身不提供这些基础设施工具；它提供的是在此类环境*中*运行的编程原语。

### 多主机环境中的JAX原语

好消息是，JAX的核心原语在设计时就考虑到了多主机场景，特别是针对SPMD（单程序，多数据）工作负载。

- **全局设备感知：** 当JAX程序在一个正确配置的多主机环境（例如TPU Pod）中运行时，`jax.devices()`将返回跨越*所有*主机的*所有*参与设备的列表。JAX会自动识别设备布局。
- **跨主机`pmap`：** `pmap`函数作用于这个全局设备列表。为单个主机编写的`pmap`代码通常可以在多主机环境中运行。你可以将函数映射到所有可用设备上，无论它们位于哪个主机。数据分片需要考虑到全局设备的总数。
- **跨主机集合操作：** `jax.lax.psum`、`pmean`、`all_gather`等集合通信原语是设备间信息交换的主要方式。在多主机设置中，这些操作透明地处理所需的网络通信。在TPU Pod等平台上，这些集合操作经过高度优化，以便使用TPU芯片之间专用的高速互连，从而减少通信开销。
- **进程识别：** JAX提供了`jax.process_index`和`jax.process_count`。`jax.process_count`会告知有多少个不同的JAX进程（通常每个主机一个）参与了计算。`jax.process_index`给出当前进程的唯一序号（通常从0到`process_count - 1`）。这些可用于，例如，让每个主机从磁盘加载大型数据集的不同分片。

> 一个`pmap`作用于两个主机（每个主机有四个设备）的视图。`pmap`将工作分配到所有八个全局设备上。集合操作（`psum`）聚合所有设备上的结果，这需要通过主机间的网络互连进行通信。

### SPMD 的扩展

单程序多数据模型自然地适用于多主机。每个主机都会启动并运行包含JAX代码的*完全相同*的Python脚本。在此脚本内部：

- 数据加载可能因主机而异，使用`jax.process_index`进行区分。
- `pmap`将函数`f`应用于分配给该主机的本地设备，对相应的数据分片进行操作。
- `pmap`内部的集合操作处理所有参与集合的主机上所有设备之间必要的数据交换，通过`axis_name`指定。

以数据并行训练为例：每个主机加载全局小批量数据的不同部分。`pmap`在每个主机的设备上本地计算梯度。`jax.lax.pmean`会沿着映射轴对梯度求平均值，跨越*所有*全局设备，在优化器步骤执行之前（优化器步骤也可能通过`pmap`执行或在所有主机上相同运行）。

### 展望

本节为多主机编程提供了基本内容。实际实施和运行多主机JAX作业涉及具体的设置步骤，这些步骤取决于你的硬件（例如TPU Pods、多节点GPU集群）以及基础设施（云平台、SLURM）。尽管针对特定环境的详细实施指南超出了我们目前的范围，但理解这些内容很重要。这展示了JAX的设计如何使用`pmap`和集合操作等一致的编程抽象，从单设备执行扩展到大型分布式系统。后面会讨论的Flax和Haiku等框架通常在这些原语之上构建，以提供更方便的API，用于在多主机设置中管理分布式训练循环和模型状态。

## 参考资料

- [Distributed JAX](https://jax.readthedocs.io/en/latest/multi_host.html) — The JAX authors (2024)
  Publisher: The JAX authors
  这份官方文档详细介绍了JAX的分布式计算方法，涵盖了全局设备管理、跨主机pmap和集合操作。
- [Cloud TPU system architecture](https://cloud.google.com/tpu/docs/system-architecture) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  提供了Google Cloud TPU的技术描述，包括其高速互连以及如何高效支持分布式训练。
- [Using MPI: Portable Parallel Programming with the Message-Passing Interface](https://mitpress.mit.edu/books/using-mpi) — William Gropp, Ewing Lusk, and Anthony Skjellum (1994)
  Publisher: The MIT Press
  一本广泛认可的书籍，解释了消息传递接口，其集合通信概念对于多主机JAX程序中的高效数据交换至关重要。
