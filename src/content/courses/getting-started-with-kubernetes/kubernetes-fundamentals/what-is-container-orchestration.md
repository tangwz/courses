---
course: "getting-started-with-kubernetes"
chapter: "kubernetes-fundamentals"
lesson: "what-is-container-orchestration"
sourceId: 7295
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/what-is-container-orchestration"
title: "什么是容器编排"
description: "了解容器编排的目的，以及为什么在大规模管理应用程序时需要像 Kubernetes 这样的系统。"
order: 1
plots: []
sourceHash: "6348ab1dbb4a0db92766a907b5ccd24677fae0c7e12ad9be2d5c1cbfafddcbcc"
sourceCorrections: []
---

虽然像 Docker 这样的容器运行时提供了构建和运行单个容器的工具，但它并不解决管理分布式应用程序的复杂挑战。当你从在一台机器上运行几个容器，转变为在服务器集群中运行成百上千个容器时，会面临一系列需要系统化自动化的新问题。这就是容器编排的职责所在。

容器编排是指对容器化应用程序整个生命周期的自动化管理。它处理诸如将容器调度到集群节点上、确保高可用性、扩缩应用程序实例以及管理它们之间的网络通信等任务。你可以将编排器看作分布式系统的“大脑”。它观察系统的当前状态，并将其与你定义的预期状态进行比较，然后采取行动使当前状态与预期状态保持一致。

### 编排解决的问题

容器编排的核心是为大规模运行应用程序时出现的几个不同运维问题提供解决方案。

#### 调度与资源管理

在拥有多台机器（称为节点）的集群中，首要任务是决定哪个节点应该运行新容器。编排器的调度器会自动做出这种放置决策。它分析容器的资源需求（如 CPU、内存），并检查每个节点的可用容量以找到合适的存放位置。这可以防止特定节点过载，并确保在无需人工干预的情况下高效使用集群的整体资源。

#### 容错与自愈

硬件会损坏，程序会崩溃。在分布式系统中，这些情况不是意外，而是常态。编排系统就是为应对这种现实而设计的。如果某个节点失去响应或某个容器进程异常终止，编排器会检测到这类故障。然后，它会自动将受影响的容器重新调度到健康的节点上，以恢复应用程序预期的运行实例数量，这一过程通常称为“自愈”。

> 人工管理下的系统在容器发生故障时需要人工介入，而编排系统会自动检测故障并替换容器，以维持预期状态。

#### 扩缩容与负载均衡

应用程序的流量很少是固定不变的。编排器允许你根据负载动态调整容器副本的数量。你可以增加副本数来应对流量高峰，并在空闲期间减少副本以节省资源。此外，当你运行应用程序组件的多个副本时，编排器会提供内部负载均衡，将传入的网络流量均匀地分配到所有可用实例上，防止任何单个容器成为瓶颈。

#### 服务发现与网络

分布式系统中的容器是临时性的。它们会被创建、销毁并在节点间移动，这意味着它们的 IP 地址经常变化。这带来了一个极大的挑战：一个容器如何可靠地找到另一个容器并进行通信？编排平台通过内置的网络模型解决了这个问题。它们提供稳定的网络端点（在 Kubernetes 术语中称为 Service）和内部 DNS 系统，允许应用程序使用固定的名称而不是易变的 IP 地址来相互发现和连接。

Kubernetes 是一个生产级的编排系统，它提供了统一的 API 和一组功能强大的组件，以声明式的方式解决这些问题。在接下来的部分中，我们将查看使 Kubernetes 能够可靠且高效地管理复杂应用程序的架构设计。

## 参考资料

- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/) — Kubernetes Authors (2020)
  官方文档全面解释了Kubernetes的核心概念、架构及其如何解决分布式应用中的调度、自愈、扩展和网络等挑战。
- [Large-scale cluster management at Google with Borg](https://doi.org/10.1145/2741948.2741964) — Abhishek Verma, Luis Pedrosa, Madhukar Korupolu, David Oppenheimer, Eric Tune, and John Wilkes (2015)
  Journal: Proceedings of the Tenth European Conference on Computer Systems (EuroSys '15); Publisher: ACM; Pages: Article 18, 1-17; DOI: [10.1145/2741948.2741964](https://doi.org/10.1145/2741948.2741964)
  这篇开创性论文描述了Google的Borg系统，它是Kubernetes的奠基性前身，阐明了大规模容器编排所面临的挑战和解决方案。
