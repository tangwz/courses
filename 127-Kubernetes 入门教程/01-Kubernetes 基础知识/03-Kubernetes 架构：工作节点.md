# Kubernetes 架构：工作节点

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/kubernetes-architecture-worker-node)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Kubernetes 控制平面相当于集群的大脑，负责做出决策并管理整体状态。相比之下，工作节点则是执行实际任务的肌肉。每个工作节点都是一台机器（无论是物理机还是虚拟机），负责运行你部署的容器化应用。为了实现这一点，每个节点都会运行一组核心组件，这些组件接收来自控制平面的指令，并管理该机器上工作负载的生命周期。

让我们看看运行在每个工作节点上的三个核心组件。

### Kubelet：节点代理

Kubelet 是运行在每个工作节点上的主要代理。它的主要职责是确保 Pod 规范（PodSpecs）中描述的容器正在运行且处于健康状态。Kubelet 不会管理那些非 Kubernetes API Server 指派给它的容器。

它的角色相当于一个通信桥梁：

- **接收指令：** Kubelet 监听 API Server，查看是否有调度到该节点的 Pod。
- **执行操作：** 当收到 PodSpec 时，它会与容器运行时进行交互，拉取所需的容器镜像并启动容器。
- **上报状态：** 它持续向控制平面反馈节点及其运行 Pod 的健康状况和状态。如果某个容器未通过健康检查，Kubelet 会根据 Pod 的重启策略将其重启。

### 容器运行时：引擎

容器运行时是负责运行容器的软件。虽然 Docker 在早期非常流行，但 Kubernetes 支持任何符合其容器运行时接口（CRI）的运行时。这种灵活性允许使用更轻量、更高效的运行时，例如 `containerd` 或 `CRI-O`。

Kubelet 与容器运行时进行通信，以处理所有容器层面的操作：

- 从镜像仓库拉取容器镜像。
- 基于该镜像启动新容器。
- 停止正在运行的容器。
- 销毁容器。

从本质上讲，Kubelet 将抽象的 Pod 定义转换为容器运行时可以执行的具体动作。

> 工作节点组件运行示意图。API Server 向 Kubelet 发送指令，Kubelet 指导容器运行时管理 Pod。Kube-proxy 则负责这些 Pod 的网络路由。

### Kube-proxy：网络管理器

Pod 之间（可能跨越多个节点）的网络通信是一项基本要求。Kubernetes 代理（即 `kube-proxy`）是一个运行在每个节点上的网络代理，它是 Kubernetes 网络模型的组成部分。

它的主要功能是在宿主操作系统上维护网络规则。这些规则通常在 Linux 上使用 `iptables` 或 `IPVS` 实现，确保网络流量能正确路由到 Pod。当你创建一个为一组 Pod 提供稳定 IP 地址的 Kubernetes Service 时，正是 `kube-proxy` 将 Service 的虚拟 IP 转换为后端 Pod 的实际 IP 地址，并在它们之间实现流量负载均衡。

这三个组件协同工作，使应用得以运行。Kubelet 充当本地监管者，容器运行时处理底层容器机制，而 `kube-proxy` 负责网络连接。这种分布式架构支持 Kubernetes 的水平扩展。通过添加更多工作节点，你可以为控制平面提供更多资源，从而在无需手动干预单台机器的情况下，让集群运行更多应用。

## 参考资料

- [Nodes](https://kubernetes.io/docs/concepts/architecture/nodes/) — Kubernetes Authors (2024)
  这份官方文档页面提供了对 Kubernetes 工作节点及其基本组件--Kubelet、容器运行时和 kube-proxy 的基础理解。
- [Container Runtime Interface (CRI)](https://kubernetes.io/docs/concepts/architecture/cri/) — Kubernetes Authors (2025)
  解释了容器运行时接口（CRI），该接口标准化了 Kubelet 与 containerd 或 CRI-O 等各种容器运行时之间的通信。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running/9781098110263/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media; Pages: 320
  这是一本广受好评的书，全面涵盖了 Kubernetes 架构，包括对工作节点组件及其交互的详细解释。第四版。

---

[上一节](02-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E6%8E%A7%E5%88%B6%E5%B9%B3%E9%9D%A2.md) · [下一节](04-kubectl%20%E7%9A%84%E4%BD%9C%E7%94%A8.md)
