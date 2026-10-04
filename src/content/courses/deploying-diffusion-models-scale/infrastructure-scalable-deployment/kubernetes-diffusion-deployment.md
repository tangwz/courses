---
course: "deploying-diffusion-models-scale"
chapter: "infrastructure-scalable-deployment"
lesson: "kubernetes-diffusion-deployment"
sourceId: 5411
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment/kubernetes-diffusion-deployment"
title: "使用 Kubernetes 进行编排"
description: "在 Kubernetes 上部署容器化扩散模型以实现可伸缩性和弹性。"
order: 3
plots: []
sourceHash: "819e61155dc2d0746826e4c0c780d5f9ef0988444a06bac92872bf9223acbe8b"
sourceCorrections: []
---

使用 Docker 将扩散模型应用容器化，可将模型、依赖项和运行时打包成一个可移植单元，这是部署的有效实践。然而，在需要可伸缩性、弹性和高效资源管理的生产环境中，运行单个容器，甚至少量手动管理的容器，很快就变得难以维持。这就是 Kubernetes 等容器编排平台发挥作用的地方。

Kubernetes 提供了一个框架，用于自动化容器化应用的部署、扩展和管理。对于计算密集型且通常长时间运行的任务，例如扩散模型推理 (inference)，Kubernetes 提供处理相关特定难题的机制。它管理跨机器集群（节点）的整个应用生命周期。

### 为什么扩散模型需要 Kubernetes？

大规模部署扩散模型带来了一些挑战，Kubernetes 有助于解决这些问题：

1. **可伸缩性：** 处理波动的用户需求需要自动增减推理 (inference)工作器的数量。Kubernetes 提供水平 Pod 自动伸缩（HPA），可以根据 CPU、内存等指标，甚至 GPU 利用率或队列长度等自定义指标（稍后讨论）来管理此过程。
2. **资源管理：** 扩散模型需要大量资源，特别是 GPU。Kubernetes 配合合适的设备插件，允许您为容器请求 GPU 资源，并确保它们调度到配备 GPU 的节点上。它有助于高效管理这些昂贵资源的分配。
3. **高可用性：** Kubernetes 可以自动重启失败的容器并将其重新调度到健康节点上，提升推理服务的容错能力。部署（Deployments）确保所需数量的副本正在运行。
4. **服务发现和负载均衡：** 随着 Pod 的创建和销毁，它们的 IP 地址会变化。Kubernetes Service 提供一个稳定端点（IP 地址和 DNS 名称）来访问您的应用，自动在可用 Pod 之间进行请求负载均衡。
5. **滚动更新和回滚：** 在不中断服务的情况下更新模型或应用代码是必需的。Kubernetes Deployments 促进滚动更新，逐步替换旧的 Pod 为新的，并在出现问题时允许快速回滚。

### 用于部署的 Kubernetes 核心组件

要在 Kubernetes 上部署您的容器化扩散模型，您主要会与这些基本对象进行交互：

- **Pod：** Kubernetes 中最小的可部署单元，代表运行进程的一个实例。一个 Pod 封装一个或多个容器（通常是您的扩散模型推理 (inference)服务器容器）、存储资源、一个唯一的网络 IP，以及管理容器运行方式的选项。
- **Deployment：** 一个更高级别的对象，管理一组相同的 Pod（副本）。您声明所需状态（例如，“运行我的推理服务器容器的 3 个副本，使用镜像 X”），然后 Deployment 控制器致力于维持该状态。Deployment 处理伸缩和更新。
- **Service：** 一种抽象，定义了一组逻辑 Pod 和访问它们的策略。Service 提供一个稳定的 IP 地址和 DNS 名称。当请求到达 Service IP 时，Kubernetes 会将其路由到关联 Deployment 管理的一个健康 Pod，有效地提供负载均衡。常见类型包括 `ClusterIP`（内部访问）、`NodePort`（在每个节点的 IP 上暴露）和 `LoadBalancer`（提供一个云负载均衡器）。
- **Namespace：** 用于在物理集群内创建虚拟集群。它们提供名称作用域，也是一种在多个用户、团队或应用之间划分集群资源的方式。

### Kubernetes 上的推理 (inference)架构设计

虽然一个简单的公开 Service 的 Deployment 可能适用于基本情况，但扩散模型推理的特性通常更适用于更精巧的架构。一个常见模式是将请求处理与 GPU 密集型推理工作解耦：

> Kubernetes 上扩散模型推理的解耦架构。传入请求由轻量级 API 前端 Pod 处理，它们将生成任务排队。专用的 GPU 工作器 Pod 从队列中消费任务，执行推理，并异步处理结果。

在这种架构中：

1. **API 前端 (Deployment)：** 轻量级 Web 服务器 Pod（例如 Flask、FastAPI）处理传入的用户请求。它们验证输入，或执行初步检查，然后将生成任务参数 (parameter)推送到消息队列（如 RabbitMQ、Redis Streams 或 AWS SQS）。这些 Pod 通常不需要 GPU，并且可以根据请求量独立扩展。
2. **消息队列：** 作为缓冲和解耦机制。它保存待处理的推理任务。
3. **GPU 推理工作器 (Deployment)：** 这些 Pod 包含扩散模型和推理逻辑。它们连接到消息队列，拉取任务，使用分配的 GPU 资源执行计算密集型的生成过程，然后存储结果（例如在云存储中）或在完成后通知另一个服务。这些工作器根据队列长度或 GPU 利用率进行扩展。

这种基于队列的方法提供更好的抗负载峰值弹性，并处理扩散推理可能较长的持续时间，而不会阻塞 API 前端或导致用户请求超时。

### 配置和 GPU 识别

Kubernetes 需要特定的配置才能有效地管理 GPU 资源：

- **资源请求/限制：** 定义 Worker Pod 时，您必须使用扩展资源表示法（例如 `nvidia.com/gpu: 1`）指定 GPU 资源。这告诉 Kubernetes 调度器只将 Pod 放置在具有所需 GPU 的节点上。
- **节点污点和容忍度 / 节点选择器：** 您可以使用节点污点来阻止非 GPU 工作负载调度到昂贵的 GPU 节点上，并为您的 GPU 工作器 Pod 添加相应的容忍度。或者，节点选择器或节点亲和性可以确保 Pod 落在具有特定标签的节点上（例如 `gpu=true`、`gpu-type=nvidia-a100`）。
- **设备插件：** Kubernetes 本身不原生管理 GPU 的具体细节。您需要在启用 GPU 的节点上安装合适的设备插件（例如 NVIDIA 设备插件）。该插件检测 GPU，将其报告给 Kubernetes，并处理 GPU 访问的容器运行时配置。我们将在下一节中阐述 GPU 节点管理。
- **ConfigMap 和 Secret：** 应用配置（如模型路径、队列连接字符串、默认采样器设置）应使用 ConfigMap 进行管理。敏感信息（如 API 密钥或数据库凭据）应使用 Secret。这些使您能够将配置与容器镜像解耦。

将您的容器化扩散模型部署到 Kubernetes 上，将您的应用从一个独立进程转变为一个可伸缩、有弹性的服务。它为管理生产环境中复杂、资源密集型的工作负载提供了必要的控制平面，为后续章节和部分中涵盖的自动伸缩、监控和更新策略的实施铺平了道路。本章稍后的实践部分将指导您在 Kubernetes 集群上部署一个基本的扩散模型服务。

## 参考资料

- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/) — Kubernetes Authors (2020)
  提供对 Kubernetes 架构以及 Pod、Deployment、Service 和 Namespace 等核心对象的理解。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running-2nd/9781492046530/) — Brendan Burns, Joe Beda, Kelsey Hightower (2019)
  Publisher: O'Reilly Media
  一本全面的指南，涵盖 Kubernetes 基础知识、部署策略和容器化应用的运维方面。
- [NVIDIA Container Toolkit - Kubernetes Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  官方文档，说明如何在 Kubernetes 集群中启用 GPU 支持，对调度和管理 GPU 密集型工作负载至关重要。
- [MLOps Engineering at Scale](https://www.oreilly.com/library/view/mlops-engineering-at/9781098135836/) — Ori Rafael, Roy Hasson, Rick Bilodeau (2023)
  Publisher: O'Reilly Media
  讨论 MLOps 的最佳实践，包括在 Kubernetes 等平台上部署机器学习推理服务的可扩展模式和架构考量。
