# Kubernetes 管理机器学习工作负载简介

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/kubernetes-for-ml-workloads)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Docker 提供了一种将机器学习 (machine learning)应用打包成可移植容器的标准化方法。然而，当您需要大规模运行这些容器时，就会出现一系列新的挑战。如何为分布式训练任务部署和管理数百个容器？即使底层机器发生故障，如何确保您的模型服务 API 始终可用？在多台服务器上手动管理这些任务效率低下且容易出错。

这正是像 Kubernetes 这样的容器编排器发挥作用的地方。Kubernetes 最初由 Google 开发，现在由云原生计算基金会 (CNCF) 维护，它能自动化容器化应用的部署、扩展和运维管理。它充当机器集群的操作系统，屏蔽底层硬件并提供统一的 API 来管理您的工作负载。您无需管理单台机器，而是管理一个共享的资源池。

### Kubernetes 为何适合机器学习 (machine learning)

对于机器学习工程师和 MLOps 专业人员来说，Kubernetes 提供了一个强大的平台，可以解决机器学习生命周期中的几个常见问题。它为实验、训练和部署提供了一致的基础。

- **应对高要求工作负载的可扩展性：** 机器学习训练任务，特别是对于大型模型，通常需要跨多台机器运行才能在合理的时间内完成。Kubernetes 可以通过一个命令扩展训练任务的容器副本数量。同样，如果您的模型服务端点遇到高流量，Kubernetes 可以自动横向扩展推理 (inference)容器的数量来处理负载，并在流量减少时自动缩减。
- **高效资源管理：** 机器学习应用是资源密集型的，通常需要特定数量的 CPU、内存，最重要的是 GPU。Kubernetes 允许您为每个工作负载定义资源请求和限制。这确保了关键训练任务获得所需的 GPU，同时防止实验性 notebook 消耗集群的所有资源。它能够对昂贵的硬件进行细粒度控制。
- **可移植性和一致性：** 正如 Docker 为单个应用提供一致性一样，Kubernetes 为整个系统提供一致性。您可以使用一组声明式配置文件（通常用 YAML 编写）定义您的完整机器学习技术栈，包括训练管道、模型服务 API、监控仪表盘等。然后，整个技术栈可以部署到任何 Kubernetes 集群上，无论它运行在本地、AWS、GCP 还是 Azure 上，从而确保您的环境在任何地方都是可复现的。
- **弹性与自修复：** 长期运行的训练任务可能被硬件故障或其他暂时性问题中断。Kubernetes 会自动监控容器的健康状况，并重启任何失败的容器。对于已部署的模型，它可以保持指定数量的运行副本，自动替换任何崩溃的副本。这种自修复能力对于构建可靠的机器学习系统非常重要。

### Kubernetes 集群抽象

使用 Kubernetes 时，基本的转变是从考虑单台机器转变为考虑一个统一的集群。您声明您希望应用程序达到的状态，Kubernetes 的控制平面会通过将工作调度到所有可用的工作机器（这些机器被称为节点）上来实现这一状态。

> 工程师在 YAML 文件中定义应用程序的期望状态，并将其发送到控制平面。控制平面的调度器随后找到合适的工作节点，在 Pods 中运行应用程序的组件，并在需要时分配 GPU 等资源。

### 工作原理：概览

您使用声明式方法与 Kubernetes 交互。您无需编写脚本来发出命令序列，而是创建声明应用程序期望状态的配置文件。例如，您可以声明：“我希望我的模型服务容器有三个副本正在运行，并且它应该可以通过公共互联网访问。” Kubernetes 会持续努力使集群的实际状态与您声明的状态匹配。

为了实现这一点，Kubernetes 构建于几个核心组件之上，我们将在下一节中详细查看它们：

- **控制平面：** 这是集群的协调大脑。它暴露 API，观察新的应用定义，并做出全局决策，例如将容器调度到机器上。
- **节点：** 这些是组成集群计算能力的工作机器，可以是物理的或虚拟的。每个节点都运行一个与控制平面通信的代理，并管理分配给它的容器。
- **Pods：** Pod 是您在 Kubernetes 对象模型中创建或部署的最小和最简单的单元。一个 Pod 封装了您的容器化应用（例如，一个包含您的模型 API 代码的 Docker 容器），以及其存储资源和唯一的网络 IP。虽然一个 Pod 可以包含多个容器，但最常见的模式是一个 Pod 一个容器。

对 Kubernetes 是什么以及为何它是适合机器学习 (machine learning)的平台有所了解后，我们现在可以查看您将用于定义和部署应用程序的具体对象。

## 参考资料

- [Kubernetes Documentation](https://kubernetes.io/docs/) — Kubernetes Authors (2024)
  Publisher: Cloud Native Computing Foundation (CNCF)
  学习 Kubernetes 核心概念、架构和 API 的权威来源。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure (4th Edition)](https://www.oreilly.com/library/view/kubernetes-up-running/9781098110598/) — Brendan Burns, Joe Beda, Kelsey Hightower (2023)
  Publisher: O'Reilly Media
  由 Kubernetes 共同创建者撰写的综合书籍，详细解释了其设计和使用。
- [Kubeflow Documentation](https://www.kubeflow.org/docs/) — Kubeflow Authors (2020)
  用于在 Kubernetes 上部署和管理机器学习工作流（包括分布式训练和模型服务）的官方文档。
- [MLOps Engineering at Scale](https://www.oreilly.com/library/view/mlops-engineering-at/9781098111977/) — Diego Oppenheimer (2022)
  Publisher: O'Reilly Media
  提供了将 Kubernetes 集成到 MLOps 实践中，以构建可扩展和可靠的机器学习系统的背景。

---

[上一节](02-%E6%9E%84%E5%BB%BA%E5%8C%85%E5%90%AB%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BA%93%E7%9A%84Docker%E9%95%9C%E5%83%8F.md) · [下一节](04-Kubernetes%20%E7%BB%84%E4%BB%B6%EF%BC%9APod%E3%80%81Service%E3%80%81Deployment.md)
