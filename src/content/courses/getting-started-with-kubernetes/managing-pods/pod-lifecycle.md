---
course: "getting-started-with-kubernetes"
chapter: "managing-pods"
lesson: "pod-lifecycle"
sourceId: 7305
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods/pod-lifecycle"
title: "Pod 的生命周期"
description: "了解 Pod 经历的不同阶段，从 Pending 到 Running、Succeeded 和 Failed，以及每个阶段所代表的含义。"
order: 4
plots: []
sourceHash: "57c46e97adce275029fbe167451e77197586bc3a4854f2aa72acf17af2a30cd0"
sourceCorrections: []
---

一旦 Pod 清单提交给 Kubernetes API 服务器，Pod 并非直接出现，而是会经历一个明确的生命周期。了解这个生命周期是管理和排查应用程序故障的基础。Kubernetes 主要通过其 `status` 对象中的 `phase` 字段来跟踪 Pod 的状态。这个阶段是对 Pod 从创建到终止所处位置的高级概括。

### Pod 阶段 (Pod Phases)

集群中的每个 Pod 都处于五个可能阶段之一。这些阶段代表了 Pod 内容器的总体状态。

- **Pending（等待中）**：Pod 已被 Kubernetes 集群接受，但其中一个或多个容器镜像尚未创建，也尚未运行。出现这种延迟可能有多种原因，例如调度程序正在等待寻找具有足够资源的合适节点，或者节点正在下载大型容器镜像。
- **Running（运行中）**：Pod 已绑定到节点，且所有容器均已创建。至少有一个容器仍在运行，或者正在启动或重启。需要注意的是，处于 `Running` 阶段的 Pod 并不一定意味着其中的应用程序是健康的或已准备好处理请求。它仅表示容器进程已经启动。
- **Succeeded（成功）**：Pod 中的所有容器均已终止，退出状态码为 0，表示执行成功。这些 Pod 不会被重启。这种阶段常见于一次性运行并退出的短时任务或批处理作业。
- **Failed（失败）**：Pod 中的所有容器均已终止，且至少有一个容器以非零状态码终止，表示失败。此阶段表明 Pod 的某个容器内发生了错误。
- **Unknown（未知）**：无法获取 Pod 的状态。这通常发生在控制平面与 Pod 运行节点上的 Kubelet 失去通信时。

下图展示了这些阶段之间的主要转换过程。

> Kubernetes Pod 的生命周期，展示了从 Pending（等待中）到 Succeeded（成功）或 Failed（失败）终止状态的过程。

### Pod 条件 (Pod Conditions)

虽然 `phase` 提供了快速概括，但有时会掩盖底层细节。为了获得更细致的信息，Kubernetes 在 Pod 的 `status` 字段中使用了一组条件（Conditions）。每个条件都是一个布尔值（`True`、`False` 或 `Unknown`），代表 Pod 状态的特定方面。

最主要的条件包括：

- **PodScheduled**：表示 Pod 是否已调度到节点。
- **Initialized**：所有 [初始化容器 (init containers)](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) 均已成功完成。
- **ContainersReady**：Pod 中的所有容器都已就绪。这与我们稍后讨论的就绪探针密切相关。
- **Ready**：Pod 能够处理请求，应添加到所有匹配 Service 的负载均衡池中。仅当 `ContainersReady` 为 `True` 时，`Ready` 条件才为 `True`。

常见的情况是 Pod 处于 `Running` 阶段，但其 `Ready` 条件为 `False`。这表明虽然容器进程处于活跃状态，但其中的应用程序可能仍在启动、加载数据或未通过健康检查。

### 容器状态 (Container States)

进一步细分，Kubernetes 还会跟踪 Pod 内每个单独容器的状态。容器可以处于以下三种状态之一：

- **Waiting（等待中）**：容器未运行。可能处于此状态的原因包括正在拉取镜像、运行启动前钩子或等待 Secret 可用。Pod 状态中会提供处于 `Waiting` 状态的原因。你可能会经常看到 `ImagePullBackOff`，这意味着 Kubernetes 在获取容器镜像时遇到困难。
- **Running（运行中）**：容器正在正常执行。
- **Terminated（已终止）**：容器已完成执行并停止。处于此状态的原因可能是成功完成了任务，也可能是发生了失败。退出代码和原因等详细信息均可查阅。反复失败并由 Kubernetes 重启的容器可能会显示原因为 `CrashLoopBackOff`。

你可以使用 `kubectl describe pod` 命令检查所有这些细节。输出结果提供了全方位的视图，包括 Pod 的当前阶段、其条件以及每个容器的状态。当你需要诊断 Pod 行为异常的原因时，这些细节信息非常有价值。

例如，观察到一个 Pod 卡在 `Pending` 状态，可能会促使你检查 `PodScheduled` 条件，该条件可能为 `False`。随后，`kubectl describe` 输出中的事件部分可能会告诉你调度器无法放置该 Pod 的原因，例如所有可用节点的 CPU 或内存不足。

## 参考资料

- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) — Kubernetes Authors (2024)
  全面解释了 Pod 的阶段、条件和容器状态。
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) — Kubernetes Authors (2024)
  提供关于 Pod、其目的和组件的基础信息。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and/9781098110192/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本广泛认可的著作，提供 Kubernetes 的实践和理论方面内容。
- [Debugging Pods](https://kubernetes.io/docs/tasks/debug-application-cluster/debug-pods/) — The Kubernetes Authors (2025)
  Publisher: The Linux Foundation
  关于如何诊断和解决 Pod 问题的指南。
