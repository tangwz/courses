---
course: "getting-started-with-kubernetes"
chapter: "networking-in-kubernetes"
lesson: "service-types-clusterip-nodeport-loadbalancer"
sourceId: 7318
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/service-types-clusterip-nodeport-loadbalancer"
title: "Service 类型：ClusterIP、NodePort 与 LoadBalancer"
description: "对比不同的 Service 类型——ClusterIP、NodePort 和 LoadBalancer，并了解每种类型的适用场景。"
order: 3
plots: []
sourceHash: "2cce51e05ce33d79c4a89688e4b541de959673bc1b393070d1eb6fd028a1c0c1"
sourceCorrections: []
---

Kubernetes Service 为一组 Pod 提供稳定的端点。然而，并非所有服务都需要以相同的方式暴露。后端 API 使用的数据库与面向公众的 Web 服务器在网络需求上大不相同。Kubernetes 通过提供不同类型的 Service 来满足这些多样化的需求，这些类型由 Service 清单中的 `spec.type` 字段控制。每种类型都提供了一种独特的访问模式。本节分析涵盖了三种主要类型：`ClusterIP`、`NodePort` 和 `LoadBalancer`。

### ClusterIP：仅限集群内部通信

`ClusterIP` 是 Kubernetes 默认的 Service 类型。当你在创建 Service 时没有明确指定 `type` 时，就会得到一个 `ClusterIP`。这种类型将 Service 暴露在一个仅在集群内部可达的内部 IP 地址上。

这是最常用的 Service 类型，非常适合用于集群内部不同微服务之间的通信。例如，Web 应用程序的前端可能需要与后端的内置用户身份验证服务通信。由于身份验证服务不应直接暴露给外部互联网，因此 `ClusterIP` Service 是理想的选择。它提供了一个可靠的内部端点供其他 Pod 访问，同时与外部流量完全隔离。

以下是一个针对带有 `app: my-api` 标签的 Pod 的 `ClusterIP` Service 最小化配置清单：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-api-service
spec:
  # type: ClusterIP 是默认值，所以这一行是可选的
  type: ClusterIP
  selector:
    app: my-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

在此配置中，集群内部的其他应用程序可以通过向 `my-api-service:80` 发送请求来访问 API Pod，Kubernetes 将处理匹配 Pod 之间的负载均衡。

> `ClusterIP` 服务提供了一个私有的虚拟 IP，使其只能被运行在同一个 Kubernetes 集群中的其他工作负载访问。

### NodePort：在每个节点上暴露服务

`NodePort` Service 类型是将应用程序暴露给外部流量的一种直接方式。当你将类型设置为 `NodePort` 时，Kubernetes 会在集群中的每个工作节点上，从预配置的范围（通常是 30000–32767）中分配一个静态端口。随后，通过访问任何节点的 IP 地址及该分配端口：`<NodeIP>:<NodePort>`，即可从集群外部访问该 Service。

当你创建一个 `NodePort` Service 时，Kubernetes 也会自动创建一个 `ClusterIP` Service。这意味着该 Service 仍然可以通过其 `clusterIP` 进行内部通信，同时也能通过节点端口暴露给外部。

这种类型通常用于开发和测试环境，或者用于不需要专用云负载均衡器的应用场景。然而，对于生产环境的 Web 服务来说，它并不常用，因为你需要管理连接到哪个节点 IP，而且如果客户端配置不当，节点宕机可能会导致访问中断。

以下是一个 `NodePort` Service 的清单示例：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-webapp-service
spec:
  type: NodePort
  selector:
    app: my-webapp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
      # nodePort: 30080 # 这是可选的。如果省略，Kubernetes 会分配一个随机端口。
```

通过此配置，你可以通过浏览器访问 `http://<任何节点IP>:30080` 来访问集群外部的 `my-webapp` 应用程序。

> 指向任何工作节点上 `NodePort` 的流量都会由 kube-proxy 转发到该 Service 的内部 `ClusterIP`，然后再路由到其中一个目标 Pod。

### LoadBalancer：与云服务商集成

`LoadBalancer` Service 类型是将应用程序暴露给互联网的标准且最可靠的方式。此类型旨在与 AWS、Google Cloud 和 Azure 等云服务商的基础设施集成。

当你创建一个类型为 `LoadBalancer` 的 Service 时，你是在请求底层云平台配置一个外部网络负载均衡器。该负载均衡器会被分配一个稳定的公网 IP 地址，并自动配置为将外部流量路由到工作节点上该 Service 的 `NodePort`。

`LoadBalancer` Service 建立在其他两种类型之上。创建它时，Kubernetes 会自动创建 `NodePort` 和 `ClusterIP` Service 以处理内部路由。云负载均衡器仅提供稳定的公共入口点。

这是在受支持的云环境中运行面向生产的应用程序的首选方法。它处理健康检查并确保流量仅发送到健康的节点，提供了比单独使用 `NodePort` 更具弹性的设置。请注意，此类型依赖于服务商，除非你安装了可以模拟负载均衡器的特定控制器（如 MetalLB），否则它在本地环境或 Kind 等本地开发环境中将无法工作。

清单非常简单：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-production-app-service
spec:
  type: LoadBalancer
  selector:
    app: my-production-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
```

在云环境中应用此清单后，Service 的状态最终会更新，包含一个外部 IP 地址，这就是应用程序的公共入口点。

> 云服务商的负载均衡器将流量引导至集群节点上的 `NodePort`，为外部客户端提供单一且稳定的公网 IP 地址。

### Service 类型总结

选择合适的 Service 类型取决于你对应用程序访问需求的理解。

| 类型 | 访问范围 | 机制 | 常见用例 |
| --- | --- | --- | --- |
| **`ClusterIP`** | 仅限内部 | 集群内部的一个虚拟 IP。 | 后端服务、数据库、内部 API。 |
| **`NodePort`** | 通过节点 IP 外部访问 | 在每个节点上开启一个静态端口。 | 开发、测试、演示或简单的服务。 |
| **`LoadBalancer`** | 通过公网 IP 外部访问 | 配置一个外部负载均衡器（云端）。 | 需要公网访问的生产 Web 应用。 |

掌握了这些 Service 类型，你就可以有效地控制集群内部和来自外部的网络流量。下一步是学习应用程序如何使用这些稳定的端点进行服务发现。

## 参考资料

- [Services, Load Balancing, and Networking](https://kubernetes.io/docs/concepts/services-networking/service/) — Kubernetes Authors (2024)
  Kubernetes 服务官方文档，概述了 `ClusterIP`、`NodePort` 和 `LoadBalancer` 等服务类型、其配置以及底层网络机制。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running/9781098110208/) — Brendan Burns, Joe Beda, Kelsey Hightower, and Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本指南，涵盖了Kubernetes概念，包括对服务类型及其在集群内外暴露应用中的作用的详细解释。
- [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action) — Marko Lukša (2017)
  Publisher: Manning Publications
  一本深入的图书，通过实践例子解释Kubernetes概念，包括对服务类型及其网络影响的全面介绍。
