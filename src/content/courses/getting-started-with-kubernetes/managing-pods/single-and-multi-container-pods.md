---
course: "getting-started-with-kubernetes"
chapter: "managing-pods"
lesson: "single-and-multi-container-pods"
sourceId: 7304
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods/single-and-multi-container-pods"
title: "单容器与多容器 Pod"
description: "对比使用单容器 Pod 与多容器 Pod 的设计模式，涵盖边车、代理和适配器。"
order: 3
plots: []
sourceHash: "99d5cf074b2d06b4ef91c2d6b4091d72eb362814e7ec61a46c46b64ed46a7bbe"
sourceCorrections: []
---

虽然一个 Pod 可以包含多个容器，但最常见的模式是单容器 Pod。这种模式将 Pod 视为单个应用程序的轻量级外壳，这与每个容器运行一个进程的理念非常吻合。这种方法提供了隔离性和直观的资源管理，足以满足各种无状态和有状态应用的需求。

以一个简单的 Web 服务器为例。它的唯一职责是处理 HTTP 流量。将其封装在单容器 Pod 中是在 Kubernetes 上部署它最清晰、最直接的方式。

以下是一个运行单个 Nginx 容器的 Pod 的最小化 YAML 清单：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-web-server
spec:
  containers:
  - name: nginx
    image: nginx:1.23
    ports:
    - containerPort: 80
```

在此定义中，`spec.containers` 字段是一个数组，但它只包含一个元素。Kubernetes 将调度此 Pod，拉取 `nginx:1.23` 镜像，并从中运行一个容器。Pod 与应用程序容器之间的这种一对一关系是大多数工作负载的基础。

### 多容器 Pod 模式

单容器模式简单高效，但无法满足所有使用场景。某些应用程序由紧密耦合的进程组成，这些进程在同一台机器上共同运行会获益。在这种情况下，多容器 Pod 就变得非常必要。

Pod 内的所有容器共享相同的网络命名空间，并且可以共享相同的存储卷。这意味着它们可以使用 `localhost` 相互通信，并且可以读写相同的文件。可以将多容器 Pod 想象成一个逻辑上的微型主机，其中每个容器都是该主机上运行的一个进程。这种同处一地的特性是几种成熟设计模式的基础。

> Pod 中的容器共享资源，从而实现边车模式，即边车容器通过访问主应用的文件和网络来增强其功能。

#### 边车 (Sidecar) 模式

最常见的多容器模式是 **边车 (sidecar)**。边车容器扩展或增强了主应用容器的功能。其目的是分担辅助任务，如日志记录、监控或代理请求，而无需更改应用程序代码。

一个典型的例子是日志转发边车。主应用将其日志写入共享卷上的文件。边车容器运行一个独立的进程（如 Fluentd 或 Logstash），该进程跟踪此日志文件并将条目转发到集中的日志服务。

以下是一个包含 Web 应用和日志转发边车的 Pod 清单：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-app-with-sidecar
spec:
  containers:
  - name: main-app
    image: busybox
    # 此命令模拟应用每 5 秒写入一次日志
    command: ["/bin/sh", "-c", "while true; do echo \"Log entry: $(date)\" >> /var/log/app.log; sleep 5; done"]
    volumeMounts:
    - name: log-volume
      mountPath: /var/log
  - name: sidecar-logger
    image: busybox
    # 此命令模拟边车读取日志
    command: ["/bin/sh", "-c", "tail -f /var/log/app.log"]
    volumeMounts:
    - name: log-volume
      mountPath: /var/log
  volumes:
  - name: log-volume
    emptyDir: {}
```

在此示例中：

1. 为 Pod 定义了一个名为 `log-volume` 的 `emptyDir` 卷。此卷在调度 Pod 时创建，在终止 Pod 时删除。
2. `main-app` 容器将此卷挂载到 `/var/log` 并将其日志写入 `app.log`。
3. `sidecar-logger` 容器也将同一个卷挂载到 `/var/log`，因此可以读取主应用写入的 `app.log` 文件。

这种关注点分离使您能够构建标准化的日志代理，该代理可以附加到任何应用程序，而无需开发人员将日志库集成到其代码中。

#### 适配器 (Adapter) 模式

**适配器 (adapter)** 模式用于标准化现有应用程序的输出或接口。边车是添加新功能，而适配器则是转换现有的输出，使其符合组织或系统范围的标准。

假设一个应用程序以私有的、非标准的格式发出指标。为了将其与 Prometheus 等需要特定暴露格式的监控系统集成，您可以部署一个适配器容器。该容器将与主应用共享卷或网络，读取私有指标，将其转换为 Prometheus 格式，并在新的网络端口上公开以便抓取。这使您能够在不修改源代码的情况下，将遗留应用或第三方应用集成到现代的可观测性堆栈中。

#### 大使 (Ambassador) 模式

**大使 (ambassador)** 模式为访问外部服务提供了统一的接口。它充当 Pod 内的本地代理，将流量从应用容器路由到正确的目的地。

例如，应用程序可能需要连接到分片数据库集群。与其在应用程序内部构建复杂的连接逻辑和发现机制，不如将应用程序配置为连接到简单的端点（如 `localhost:6379`）。监听该端口的大使容器将接收请求，并智能地将其代理到相应的数据库分片。如果数据库拓扑发生变化，只需更新大使的配置，而无需更改应用程序代码。这简化了应用开发并集中了连接管理。

## 参考资料

- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) — Kubernetes Authors (2024)
  提供对Pod作为Kubernetes中最小可部署单元的基本理解，包括单容器和多容器结构及其共享资源。
- [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/) — Kubernetes Authors (2024)
  阐述了卷如何在Pod内的容器之间以及容器与容器之间实现数据共享和持久化，这是Sidecar等多容器模式的必要机制。
- [Designing Distributed Systems: Patterns and Paradigms for Greenﬁeld Applications](https://www.oreilly.com/library/view/designing-distributed-systems/9781491983645/) — Brendan Burns (2018)
  Publisher: O'Reilly Media; Pages: 166
  介绍重要的云原生设计模式，包括Sidecar、Adapter和Ambassador，阐明它们如何在Pod等共享环境中实现辅助进程与主应用程序的协同部署。
