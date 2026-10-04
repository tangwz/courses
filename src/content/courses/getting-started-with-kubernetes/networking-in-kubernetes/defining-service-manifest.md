---
course: "getting-started-with-kubernetes"
chapter: "networking-in-kubernetes"
lesson: "defining-service-manifest"
sourceId: 7319
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/defining-service-manifest"
title: "定义 Service 清单"
description: "在 YAML 中编写 Service 清单的指南，包括指定选择器、端口和所需的 Service 类型。"
order: 4
plots: []
sourceHash: "34ccdd2c7aaf35ea08f447bab304717f54d56a9605ab7d69443f9fbacee2de69"
sourceCorrections: []
---

Service 清单定义了应用程序的网络端点。通过这个声明式的 YAML 文件，您可以描述一个由 Kubernetes 维护的稳定 IP 地址和 DNS 名称，即使后端 Pod 被创建、销毁或重新调度，它也能自动将流量路由到正确的 Pod 集合。

Service 清单遵循 Kubernetes 对象的标准结构，包含四个顶层字段：`apiVersion`、`kind`、`metadata` 和 `spec`。

- `apiVersion`：对于 Service，该值始终为 `v1`。
- `kind`：必须设置为 `Service`。
- `metadata`：包含有助于唯一标识 Service 对象的数据，例如其名称 `name`。
- `spec`：您在此处定义 Service 的预期特征，包括它如何查找 Pod 以及公开哪些端口。

让我们看一个基础的 `ClusterIP` Service 清单，这是默认类型，用于在集群内部 IP 上公开服务。

```yaml
# service-clusterip-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-backend-service
spec:
  type: ClusterIP # 这是默认值，可以省略
  selector:
    app: my-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

### Service 规范 (`spec`)

`spec` 字段是 Service 清单的核心，包含了实现网络抽象的配置。`spec` 中最主要的两个字段是 `selector` 和 `ports`。

#### 使用选择器将 Service 连接到 Pod

`selector` 字段是将 Service 链接到一组 Pod 的机制。它包含一组键值对，必须与您要指向的 Pod 上的标签匹配。当请求到达 Service 的 IP 地址和端口时，Kubernetes 会查阅此选择器，识别所有带有匹配标签的运行中 Pod，并将流量转发到其中一个 Pod。

这种标签选择器模式是 Kubernetes 的一项基本设计原则。它实现了 Service 与 Pod 的解耦。Service 不需要知道 Pod 的具体 IP 地址（这些地址是暂时的），只需要知道标识该组 Pod 的标签即可。

> Service 上的选择器（`app: my-backend`）确保它只将流量引导至具有相应标签的 Pod，从而为动态的 Pod 组创建了一个稳定的端点。

#### 定义网络端口

`ports` 字段是一个对象数组，每个对象定义了 Service 的特定端口映射。这允许单个 Service 公开多个端口。数组中的每个条目必须指定 `port` 和 `targetPort`。

- **`port`**：这是 Service 在其自身虚拟 IP（`ClusterIP`）上公开的端口号。集群内的其他应用程序将使用此端口连接到 Service。在我们的示例中，其他 Pod 将通过端口 `80` 连接到 `my-backend-service`。
- **`targetPort`**：这是目标 Pod 内部容器中应用程序监听的端口。到达 Service `port` 的流量将被转发到所选 Pod 上的此 `targetPort`。在示例中，发送到 Service 端口 `80` 的流量会被发送到 Pod 内部的端口 `8080`。`targetPort` 可以是数字，也可以是 Pod 规范中定义的端口名称，这种做法可以让配置更具灵活性。
- **`protocol`**：指定网络协议，默认为 `TCP`。`UDP` 和 `SCTP` 也是有效的选项。

### 不同 Service 类型的清单示例

Service 的行为主要由其 `spec` 中的 `type` 字段控制。虽然您已经了解了不同的类型，但通过清单查看它们的定义会使它们的用途更加明确。

#### 示例：NodePort Service

`NodePort` Service 在每个工作节点的 IP 地址上的静态端口上公开应用程序。这对于在开发期间公开应用程序或对于不需要云负载均衡器的服务非常有用。

```yaml
# service-nodeport-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-webapp-service
spec:
  # 在每个节点的 IP 上的静态端口公开此 Service。
  type: NodePort
  selector:
    app: my-webapp
  ports:
  - protocol: TCP
    # Service 内部 ClusterIP 上的端口
    port: 80
    # 容器上的端口
    targetPort: 8000
    # 节点 IP 上的静态端口。如果不指定，Kubernetes 会分配一个。
    nodePort: 30080
```

使用此清单，您可以通过访问 `http://<任意节点IP>:30080` 从集群外部访问您的 Web 应用程序。

#### 示例：LoadBalancer Service

`LoadBalancer` Service 是在云环境中向互联网公开应用程序的标准方式。当您创建此类 Service 时，云提供商与 Kubernetes 的集成将自动配置外部负载均衡器并为其分配公共 IP 地址。

```yaml
# service-loadbalancer-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-api-gateway
spec:
  # 配置外部负载均衡器（需要云提供商支持）。
  type: LoadBalancer
  selector:
    component: api-gateway
  ports:
  - protocol: TCP
    # 外部负载均衡器公开的端口。
    port: 80
    # 目标 Pod 上的端口。
    targetPort: 8080
```

在 AWS、GCP 或 Azure 等云环境中应用此清单后，系统将创建一个外部负载均衡器。发送到其公共 IP 端口 `80` 的流量将被路由到您端口 `8080` 上的 `api-gateway` Pod。

通过掌握 Service 清单，您可以精确控制应用程序在集群内的网络身份。您可以定义稳定的、可被发现的端点，实现微服务之间的解耦，使它们能够在动态环境下可靠地通信。

## 参考资料

- [Services, Load Balancing, and Networking](https://kubernetes.io/docs/concepts/services-networking/service/) — The Kubernetes Authors (2024)
  Kubernetes Service的权威指南，详细阐述了其目的、各种类型（ClusterIP、NodePort、LoadBalancer）以及如何使用清单文件进行配置。
- [Service v1 Core](https://kubernetes.io/docs/reference/kubernetes-api/service-resources/service-v1/) — The Kubernetes Authors (2024)
  Kubernetes Service对象的官方API参考，提供了所有字段及其类型的精确定义，是准确创建清单的重要依据。
