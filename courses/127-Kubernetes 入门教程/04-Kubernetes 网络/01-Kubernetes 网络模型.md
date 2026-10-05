# Kubernetes 网络模型

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/kubernetes-networking-model)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了构建可靠的分布式系统，各个组件必须能够稳定地相互通信。在 Kubernetes 环境中，Pod 随时可能被创建、销毁或重新调度到不同的节点上，因此依赖它们的临时 IP 地址进行通信并不是一个可持续的策略。为了解决这个问题，Kubernetes 建立了一套基础网络规则，用于管理集群内资源的交互方式。这种网络模型为您的应用提供了一个一致且可预测的环境，屏蔽了底层基础设施的复杂性。

### 每个 Pod 一个 IP 的基础

Kubernetes 网络模型最核心的原则是：**为每个 Pod 分配一个唯一的 IP 地址**。从 Pod 内部运行的容器角度来看，它们处于自己隔离的网络环境中。它们可以绑定到 `localhost` 地址上自选的任何端口，并将 Pod 的 IP 地址视为自己的地址。

这种模型与传统的容器网络方法（如默认的 Docker 桥接网络）有显著不同，后者通常要求您将端口从容器映射到宿主机。通过给每个 Pod 一个真实的 IP 地址，Kubernetes 消除了这一层复杂操作。Pod 的行为非常像网络上的虚拟机或物理主机；它拥有一个 IP 地址，其他 Pod 可以直接使用该地址与其通信。

这种设计选择简化了应用的开发和部署。您不再需要协调运行在同一台主机上的不同应用之间的端口占用问题，应用也可以从虚拟机迁移到容器，而无需对网络配置进行大的改动。

### 扁平的全集群网络

在“每个 Pod 一个 IP”的基础上，Kubernetes 强制要求一个满足以下三个条件的扁平网络空间：

1. **所有 Pod 都可以与其他所有 Pod 通信，而无需使用网络地址转换 (NAT)**。一个节点上的 Pod 只需使用 IP 地址即可连接到集群中任何其他节点上的 Pod。
2. **所有节点都可以与所有 Pod 通信（反之亦然），且无需 NAT**。这对于管理和监控任务非常有用，允许像 kubelet 这样的系统组件访问它们管理的 Pod。
3. **Pod 看到的自己的 IP 地址与其他 Pod 看到的其 IP 地址相同**。在集群网络中不存在地址转换或隐藏的情况。

这创造了一个清晰、直接的网络环境。如果您的应用运行在 IP 为 `10.244.1.5` 的 Pod 中，它可以直接与 IP 为 `10.244.2.10` 的 Pod 中的另一个应用建立连接，无论这些 Pod 物理上运行在集群的哪个位置。

> 不同节点上的 Pod 使用其唯一的 IP 地址直接通信。底层网络设施处理连接它们所需的路由。

### 容器网络接口 (CNI) 的作用

您可能会好奇这个扁平网络实际上是如何创建的。Kubernetes 本身并不实现网络层。相反，它依赖于一种称为**容器网络接口 (CNI)** 的标准插件规范。

当您搭建 Kubernetes 集群时，必须选择并安装一个 CNI 插件。该插件负责具体的操作：

- 为每个新 Pod 分配 IP 地址。
- 在每个节点上配置网络路由，以确保 Pod 可以在整个集群中相互访问。

常见的 CNI 插件包括 Calico、Flannel、Weave Net 和 Cilium。每种插件使用不同的技术（如叠加网络或 BGP）来实现 Kubernetes 网络模型，但它们都遵循相同的规则。这使您可以根据性能、安全性和运行需求选择最合适的网络提供商，而无需更改应用的编写方式。

### 剩下的挑战：临时 IP 与发现

该模型为通信提供了强大的基础。每个 Pod 都有一个唯一的、可路由的 IP 地址。然而，这也引出了一个新问题。

在前面的章节中，我们已经了解到 Pod 是临时性的。Deployment 控制器可以随时销毁一个 Pod 并创建一个新 Pod 来替换它。当新 Pod 创建时，它会被分配一个*新的* IP 地址。

如果您有一个需要与后端 API 通信的前端应用，您不能将后端 Pod 的 IP 地址写死。因为那个 IP 地址并不稳定。前端如何找到健康后端 Pod 当前有效的 IP 地址呢？

这个**服务发现**问题正是 Kubernetes `Service` 对象要解决的。虽然网络模型为我们提供了连通性，但我们需要更高层次的抽象来为一组不断变化的 Pod 提供一个稳定的端点。在下一节中，您将学习 Service 是如何实现这一点的。

## 参考资料

- [Kubernetes Concepts - Networking](https://kubernetes.io/docs/concepts/services-networking/) — Kubernetes Authors (2024)
  提供了Kubernetes网络模型的官方解释，包括IP-per-Pod和集群范围连接规则。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and/9781098110192/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media; Pages: 320
  一本权威书籍（第三版），解释了Kubernetes网络模型及其实现原理。

---

[上一节](../03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E4%B8%8E%E6%9B%B4%E6%96%B0%20Deployment.md) · [下一节](02-Service%20%E7%AE%80%E4%BB%8B.md)
