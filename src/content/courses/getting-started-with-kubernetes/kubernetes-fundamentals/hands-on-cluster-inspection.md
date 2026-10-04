---
course: "getting-started-with-kubernetes"
chapter: "kubernetes-fundamentals"
lesson: "hands-on-cluster-inspection"
sourceId: 7301
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/hands-on-cluster-inspection"
title: "动手实践：集群检查"
description: "动手实践练习，配置 kubectl 并运行命令以检查新创建的本地集群的状态和组件。"
order: 7
plots: []
sourceHash: "3f031ed5f926cb87fc2d2c17493bad28811aaad9bfa42e393532c020aa67e40a"
sourceCorrections: []
---

Kubernetes 架构由控制平面和工作节点组成。你将使用 `kubectl` 命令行工具连接到本地集群，并直接检查这些组件。通过观察运行状态中体现的系统架构，这次练习将巩固你的理解。

在开始之前，请确保使用 Minikube 或 Kind 搭建的本地 Kubernetes 集群正在运行。你的 `kubectl` 命令行工具也应配置好并能与其通信。

### 验证连接

首先，让我们确认 `kubectl` 已指向正确的集群。上下文（Context）决定了你的命令发送到哪个集群。你可以通过以下命令检查当前的上下文：

```sh
kubectl config current-context
```

输出应与你本地集群的名称匹配，例如 `minikube` 或 `kind-kind`。如果你管理多个集群，`kubectl config get-contexts` 会列出所有可用选项。

确认上下文后，运行 `kubectl cluster-info` 获取集群端点的简要摘要。

```sh
kubectl cluster-info
```

你应该会看到类似这样的输出：

```
Kubernetes control plane is running at https://127.0.0.1:50937
CoreDNS is running at https://127.0.0.1:50937/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

此输出验证了两件即：`kubectl` 可以成功与 Kubernetes API 服务器（控制平面）通信，且集群的内部 DNS 服务（CoreDNS）处于活跃状态。API 服务器端点是所有集群管理操作的入口。

### 检查集群节点

一个 Kubernetes 集群由一个或多个节点组成。在我们的本地设置中，通常只有一个节点，既充当控制平面又充当工作节点。让我们列出集群中的节点。

```sh
kubectl get nodes
```

你也可以使用较短的别名 `no`：`kubectl get no`。输出提供了一个简洁的表格：

```
NAME                 STATUS   ROLES           AGE   VERSION
minikube             Ready    control-plane   12m   v1.28.3
```

以下是各列的说明：

- **NAME:** 节点的唯一标识符。
- **STATUS:** 表示节点是否健康并准备好接收 Pod。`Ready` 是理想状态。
- **ROLES:** 显示节点的角色。在大型集群中，你会看到 `worker` 角色，但在许多本地设置中，单个节点被指定为 `control-plane`。
- **AGE:** 节点已运行的时间。
- **VERSION:** 节点上运行的 Kubernetes 版本。

### 查看 `kube-system` 命名空间

Kubernetes 使用命名空间（Namespace）来组织集群内的对象。可以将它们视为物理集群中的虚拟集群。构成 Kubernetes 本身的组件运行在名为 `kube-system` 的特殊命名空间中。

要查看作为 Pod 运行的控制平面和其他系统级组件，请执行以下命令：

```sh
kubectl get pods -n kube-system
```

`-n` 参数 (parameter)指定了命名空间。输出显示了集群运行的核心部分：

```
NAME                               READY   STATUS    RESTARTS   AGE
coredns-5dd5756b68-7v7j9           1/1     Running   0          15m
etcd-minikube                      1/1     Running   0          15m
kube-apiserver-minikube            1/1     Running   0          15m
kube-controller-manager-minikube   1/1     Running   0          15m
kube-proxy-n2g2c                   1/1     Running   0          15m
kube-scheduler-minikube            1/1     Running   0          15m
storage-provisioner                1/1     Running   0          15m
```

此列表与我们之前讨论的架构直接对应：

- `etcd-minikube`: 集群的后端存储。
- `kube-apiserver-minikube`: `kubectl` 与其通信的 API 服务器。
- `kube-controller-manager-minikube`: 运行控制器循环。
- `kube-scheduler-minikube`: 将 Pod 分配给节点。
- `kube-proxy-...`: 运行在每个节点上的网络代理。
- `coredns-...`: 为集群提供服务发现和 DNS 功能。

> **Kubelet 在哪里？**
> 你可能会注意到列表中没有 Kubelet。这是因为 Kubelet 不是以 Pod 形式运行的。它是一个系统代理，直接运行在每个节点的宿主操作系统上，通常作为 `systemd` 服务。它的职责是与容器运行时和 API 服务器通信，以管理调度到该节点的 Pod。

> 在 `kube-system` 命名空间中作为 Pod 运行的主要组件图示。

### 使用 `describe` 获取详细信息

`get` 命令提供摘要，但要详细查看任何 Kubernetes 对象，需要使用 `describe` 命令。让我们检查节点以查看更详细的信息。请将 `minikube` 替换为 `kubectl get nodes` 命令中显示的实际节点名称。

```sh
kubectl describe node minikube
```

输出内容很多，但对诊断非常有用。它包括：

- **标签（Labels）和注解（Annotations）：** 用于组织和选择对象的键值元数据。
- **系统信息：** 有关节点操作系统、内核版本和容器运行时的详细信息。
- **状态（Conditions）：** 节点的健康状况，如 `Ready`、`MemoryPressure` 或 `DiskPressure`。`Ready` 状态是节点健康运行的最主要指标。
- **容量（Capacity）和可分配资源（Allocatable）：** 有关节点总资源（CPU、内存、存储）及可供 Pod 使用的资源量的信息。
- **Pods:** 当前在节点上运行的所有 Pod 列表。
- **事件（Events）：** 与节点相关的按时间顺序排列的日志，例如它在集群中的注册。这通常是排查问题时的首选查看位置。

通过运行这些命令，你已经成功与 Kubernetes 集群的 API 进行了交互，列出了其主要资源，并将运行中的组件与本章讨论的架构联系起来。你已经确认集群可以正常运行，并准备好部署应用程序，这正是我们在下一章要做的内容。

## 参考资料

- [kubectl Overview](https://kubernetes.io/docs/reference/kubectl/overview/) — Kubernetes Authors (2024)
  提供了对 kubectl 命令行工具、其语法和常见操作的全面介绍，对于集群检查至关重要。
- [Kubernetes Components](https://kubernetes.io/docs/concepts/overview/components/) — Kubernetes Authors (2025)
  Publisher: The Kubernetes Project
  解释了 Kubernetes 集群的架构，详细介绍了控制平面和工作节点组件，如 API 服务器、etcd、调度器、控制器管理器、kubelet 和 kube-proxy，这些组件在检查过程中被识别。
- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications
  这是一本权威书籍，提供了 Kubernetes 概念、架构以及用于集群管理和检查的 kubectl 实际用法的深入解释。
- [Debugging Kubernetes Clusters](https://kubernetes.io/docs/tasks/debug/debug-cluster/troubleshooting-cluster/) — Kubernetes Authors (2025)
  提供诊断和排查 Kubernetes 集群问题的实用指南和 kubectl 命令，直接支持本节的实践检查方面。
