---
course: "getting-started-with-kubernetes"
chapter: "kubernetes-fundamentals"
lesson: "kubernetes-architecture-control-plane"
sourceId: 7296
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/kubernetes-architecture-control-plane"
title: "Kubernetes 架构：控制平面"
description: "从技术角度分析 Kubernetes 控制平面的组件，包括 API 服务器、etcd、调度器和控制器管理器。"
order: 2
plots: []
sourceHash: "38274ec2b285edc3959d35ed6360d1704a827ac0d008ac9a87725fd1d1d38186"
sourceCorrections: []
---

每个 Kubernetes 集群的核心都是控制平面。可以把它看作操作的中枢，负责管理集群状态、做出调度决策以及响应事件。它确保您的应用程序和集群本身按照您预期的状态运行。控制平面不直接运行您的应用程序容器；该任务交由工作节点完成。相反，它从中心指挥点对它们进行编排。

控制平面不是一个单一的进程，而是多个协同工作的独立组件集合。虽然您通常通过单一的 API 与集群交互，但这些组件就在幕后工作，每个组件都有特定的职责。让我们看看构成控制平面的主要组件。

### API 服务器 (kube-apiserver)

API 服务器是 Kubernetes 控制平面的入口。与集群的所有交互，无论是来自运行 `kubectl` 的用户、脚本还是另一个控制平面组件，都要经过 API 服务器。它充当网关，执行三个主要功能：

1. **认证与授权：** 它验证客户端的身份，并检查他们是否有权限执行所请求的操作。
2. **校验：** 它对传入的请求进行校验，以确保其格式正确并符合 Kubernetes 对象模型的规则。例如，它会拒绝指定了无效容器镜像名称的 Pod 定义。
3. **状态管理：** 它是唯一直接与 `etcd` 通信的组件。它从 `etcd` 读取集群的当前状态，并在验证请求后，将新的期望状态写回其中。

由于它是所有通信的中心枢纽，API 服务器被设计为一个无状态、可水平扩展的 REST 服务。所有修改集群状态的请求在持久化之前都会在此处理。

### etcd

如果说 API 服务器是入口，那么 `etcd` 就是集群的官方记录本。它是一个一致且高可用的键值存储，用作 Kubernetes 所有集群数据的后端存储。您在 Kubernetes 中创建的每个对象（例如 Pod、Deployment 或 Service）都表现为 `etcd` 中的一个条目。

它是整个集群的单一事实来源。当您向 API 服务器查询 Pod 的状态时，API 服务器会从 `etcd` 中检索该信息。当调度器决定将 Pod 放置在特定节点上时，它会通知 API 服务器，然后 API 服务器将该决定记录在 `etcd` 中。

对于生产集群，`etcd` 通常以 3 或 5 个节点的集群形式运行，以确保高可用性和数据持久性。这种冗余设计可以防止在单个控制平面节点发生故障时丢失集群状态。

### 调度器 (kube-scheduler)

调度器有一项特定的职责：它决定哪个工作节点应该运行新创建的 Pod。它实际上并不运行容器；它只负责做出位置决策。

调度过程涉及对每个需要调度的 Pod 执行两步操作：

1. **过滤：** 调度器首先为 Pod 找到一组可用的节点。它会过滤掉任何无法满足 Pod 要求的节点。这些要求可能包括 CPU 和内存请求、节点标签，或者规定 Pod 应如何共同放置的亲和性与反亲和性规则。
2. **打分：** 过滤之后，如果有多个合适的节点，调度器会通过为每个节点评分来对剩余节点进行排序。这种评分基于一组优先级函数，例如，可能倾向于运行 Pod 较少的节点，或者本地已缓存所需容器镜像的节点。得分最高的节点会被选中。

决策完成后，调度器会通知 API 服务器，API 服务器随后在 `etcd` 中用分配的节点名称更新 Pod 的定义。然后，目标节点上的 Kubelet 接管并运行该 Pod。

### 控制器管理器 (kube-controller-manager)

控制器管理器是主动将集群的“当前状态”推向“期望状态”的组件。它是一个包含多个控制器进程的单一二进制文件，每个进程负责管理集群的一个特定方面。

每个控制器都按照“调解循环”模式运行：

1. 它监控 API 服务器中其关心的对象更改。
2. 它将对象指定的期望状态与其当前的实际状态进行比较。
3. 它采取行动使当前状态与期望状态相匹配。

例如，**ReplicaSet 控制器**监控 ReplicaSet 对象。如果一个 ReplicaSet 规定它应该有三个 Pod 副本，但控制器只看到两个在运行，它将与 API 服务器通信以创建第三个副本。如果它看到四个，它将终止一个。

捆绑在控制器管理器中的其他组件包括：

- **节点控制器：** 监控工作节点的健康状况，并在节点停止响应时采取行动。
- **Deployment 控制器：** 通过创建和管理底层的 ReplicaSet 来管理新应用程序版本的发布。
- **Service 控制器：** 将 Service 连接到相应的 Pod，并在云环境中配置云提供商的负载均衡器。

> Kubernetes 控制平面内的通信流程。API 服务器作为中心网关，协调所有其他组件之间的交互，并将集群状态持久化在 etcd 中。

### 云控制器管理器 (cloud-controller-manager)

在许多现代部署中，Kubernetes 运行在 AWS、Google Cloud 或 Azure 等云提供商上。为了与这些云平台的特定功能集成，Kubernetes 使用了**云控制器管理器**。

此组件运行特定于云提供商的控制器。例如，如果您在云环境中创建一个 `LoadBalancer` 类型的 Service，云控制器管理器中的一个控制器将与云提供商的 API 交互，以配置实际的负载均衡器并将其设置为将流量路由到您的 Pod。通过将此逻辑从核心的 `kube-controller-manager` 中分离出来，Kubernetes 保持了云中立性，同时允许与底层基础设施紧密结合。

这些组件共同构成了一个灵活且可扩展的系统，用于管理容器化应用程序。虽然您大部分时间都会通过 `kubectl` 与 API 服务器交互，但了解调度器、`etcd` 和控制器管理器的作用，为解决问题和设计有效的应用程序部署提供了稳固的基础。

## 参考资料

- [Kubernetes Components](https://kubernetes.io/docs/concepts/overview/components/) — The Kubernetes Authors (2023)
  此官方文档页面详细解释了每个 Kubernetes 控制平面组件及其作用。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and/9781492046492/) — Brendan Burns, Joe Beda, Kelsey Hightower (2019)
  Publisher: O'Reilly Media
  一本广受认可的书籍，全面解释了 Kubernetes 架构，包括控制平面的内部工作原理。
