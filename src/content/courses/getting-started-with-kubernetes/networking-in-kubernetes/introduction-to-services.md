---
course: "getting-started-with-kubernetes"
chapter: "networking-in-kubernetes"
lesson: "introduction-to-services"
sourceId: 7317
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/introduction-to-services"
title: "Service 简介"
description: "了解 Service 如何为一组 Pod 提供稳定端点，从而实现可靠的通信和服务发现。"
order: 2
plots: []
sourceHash: "ad8d118d595d2bdcc19560d4f1469e68fafef6e2c1dcf5cc51a6c8338a9a3473"
sourceCorrections: []
---

在 Kubernetes 这样的动态环境中，依靠单个 Pod 的 IP 地址进行通信并不可靠。由 Deployment 管理的 Pod 随时可能被终止并替换，并在重启时获得新的 IP 地址。这使得 Pod 之间的直接通信变得脆弱且难以维护。如果前端应用需要与后端 API 通信，它不能简单地硬编码后端 Pod 的 IP 地址。

为了解决这个问题，Kubernetes 提供了一种称为 Service 的网络抽象。Service 定义了一组逻辑 Pod 以及访问它们的策略。它作为一个稳定的网络端点，代表一组功能相同的 Pod。创建 Service 时，它会被分配一个虚拟 IP 地址，称为 `ClusterIP`，该地址在 Service 的生命周期内保持不变。集群内的应用可以可靠地连接到这个 `ClusterIP`，Kubernetes 会确保请求被路由到 Service 后端健康的 Pod 之一。

### 标签（Labels）与选择器（Selectors）的作用

Service 与其目标 Pod 之间的连接不是基于静态 IP 列表，而是基于使用标签和选择器的动态查询。这种解耦是 Kubernetes 运行方式的基础。

1. **标签 (Labels)**：您可以为 Pod 分配标签（简单的键值对），用以识别其属性。例如，您可以为应用的所有后端 Pod 打上 `app: my-api` 和 `tier: backend` 的标签。
2. **选择器 (Selectors)**：Service 定义中包含一个选择器，用于指定要匹配的标签。Service 会持续扫描带有与其选择器匹配标签的 Pod。

任何带有匹配标签的 Pod 在创建时都会自动添加到 Service 的端点池中。相反，如果 Pod 被终止或其标签发生变化，它就会被移除。这种机制确保了 Service 始终拥有健康、可用 Pod 的最新列表，无需任何人工干预即可发送流量。

> 客户端 Pod 与稳定的 Service IP 进行通信。Service 使用标签选择器 (`app=backend`) 来识别并将流量转发到其中一个临时的后端 Pod。

### Service 作为内部负载均衡器

除了提供稳定端点外，Service 还充当内部负载均衡器。当流量到达 Service 的 `ClusterIP` 时，运行在每个节点上的 `kube-proxy` 组件会拦截请求并将其分发到后端 Pod 之间。这种分发通常使用轮询算法，确保请求均匀地分布在可用的副本上。

这种内置的负载均衡功能提供了可扩展性和弹性。如果您通过缩放 Deployment 来增加更多 Pod，Service 会自动将它们纳入负载均衡循环中。如果某个 Pod 未通过健康检查，Service 会将其从端点池中移除，从而防止流量被发送到不健康的实例。

这种核心机制提供了可靠的内部通信。不过，Service 暴露应用的方式会有所不同。Kubernetes 提供了不同的 Service 类型来处理不同的访问模式，从纯内部通信到向互联网公开应用，我们接下来将讨论这些内容。

## 参考资料

- [Service](https://kubernetes.io/docs/concepts/services-networking/service/) — Kubernetes Authors (2024)
  官方文档，全面概述了Kubernetes服务，包括其目的、类型以及它们如何与标签和选择器配合使用。
- [Kubernetes Up and Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and/9781098110192/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本权威书籍，解释了Kubernetes核心概念，包括专门介绍服务、标签、选择器和内部负载均衡的部分。
- [Cloud Native DevOps with Kubernetes: Building, Deploying, and Scaling Modern Applications in the Cloud](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpYyoRYTF5AgtS_fPxKlMEsgc4V-mXFHu1DHOVZ3dgmieU5YUnFxGQHu4_pVra_zVM9GhEQ2PIca2DzpInR_hdx-NiY_KjYXImCqvoaVzThlaCupXeKQcG8qN6dwJr3tmWXwu2bsdoMFAydy-_sCI3plbfqvAMBIwH582Q50Akq95kdMIfLY8c_t7M_aA6Iaabfi5IHGcxgA==) — John Arundel, Justin Domingus (2022)
  Publisher: O'Reilly Media
  提供了在Kubernetes上构建和操作应用程序的实用见解，其中专门的章节解释了Kubernetes网络，包括服务和`kube-proxy`。
