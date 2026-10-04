---
course: "getting-started-with-kubernetes"
chapter: "workload-management-controllers"
lesson: "inspecting-deployment-status"
sourceId: 7314
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/inspecting-deployment-status"
title: "检查 Deployment 状态"
description: "使用 kubectl 检查 Deployment 的状态，监控更新进度并查看其历史记录。"
order: 6
plots: []
sourceHash: "1249f3bf79eee3d6c4156cf39c3115103707a1196ac1ed6e8fe6d373e5cb83de"
sourceCorrections: []
---

当在 Kubernetes 中触发滚动更新或创建新的 Deployment 时，验证过程是否按预期进行是一项主要的管理任务。Kubernetes 提供了几个 `kubectl` 命令来检查 Deployment 的状态、监控更新进度以及查看修订历史。掌握这些命令对于管理应用生命周期和排除故障非常有用。

### 使用 `kubectl get` 进行概览状态检查

检查 Deployment 状态最快的方法是使用 `kubectl get deployments` 命令。这会提供当前命名空间中所有 Deployment 的摘要信息。

```sh
$ kubectl get deployments
NAME               READY   UP-TO-DATE   AVAILABLE   AGE
nginx-deployment   3/3     3            3           15m
```

让我们分析一下输出结果中的各列：

- **NAME**：Deployment 对象的名称。
- **READY**：显示处于就绪状态的当前副本数与期望副本数的比例。在此示例中，`3/3` 表示所有三个期望的副本都在运行并通过了就绪探针检查。
- **UP-TO-DATE**：已更新至 Deployment 模板中定义的最新版本的副本数量。在滚动更新期间，随着新 Pod 的创建，此数字会逐渐增加。
- **AVAILABLE**：可用于处理流量的副本数量。可用副本是指正在运行且就绪时间至少达到 `minReadySeconds`（如果已指定）的副本。
- **AGE**：自 Deployment 创建以来经过的总时间。

如果启动了滚动更新，可以使用 `-w` 或 `--watch` 标志来实时观察这些值的变化。

```sh
$ kubectl get deployment nginx-deployment --watch
NAME               READY   UP-TO-DATE   AVAILABLE   AGE
nginx-deployment   3/3     3            3           15m
nginx-deployment   2/3     3            2           16m
nginx-deployment   3/4     3            3           16m
nginx-deployment   4/4     4            4           16m
nginx-deployment   3/3     3            3           17m
```

### 使用 `kubectl describe` 进行详细检查

要查看 Deployment 状态的更多细节，请使用 `kubectl describe deployment` 命令。此命令提供详尽的信息，包括标签、注解、副本计数、滚动更新策略以及最近事件的日志。

```sh
$ kubectl describe deployment nginx-deployment
```

输出内容比较详尽，但以下几个部分对检查尤为重要：

```sh
Name:                   nginx-deployment
Namespace:              default
CreationTimestamp:      ...
Labels:                 app=nginx
Annotations:            deployment.kubernetes.io/revision: '2'
Selector:               app=nginx
Replicas:               3 desired | 3 updated | 3 total | 3 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
...
Events:
  Type    Reason             Age    From                   Message
  ----    ------             ----   ----                   -------
  Normal  ScalingReplicaSet  5m3s   deployment-controller  Scaled up replica set nginx-deployment-7b8d4f5c6f to 3
  Normal  ScalingReplicaSet  1m20s  deployment-controller  Scaled up replica set nginx-deployment-6c7b5d6f8g to 1
  Normal  ScalingReplicaSet  1m18s  deployment-controller  Scaled down replica set nginx-deployment-7b8d4f5c6f to 2
  Normal  ScalingReplicaSet  1m18s  deployment-controller  Scaled up replica set nginx-deployment-6c7b5d6f8g to 2
  Normal  ScalingReplicaSet  1m16s  deployment-controller  Scaled down replica set nginx-deployment-7b8d4f5c6f to 1
  Normal  ScalingReplicaSet  1m16s  deployment-controller  Scaled up replica set nginx-deployment-6c7b5d6f8g to 3
  Normal  ScalingReplicaSet  1m14s  deployment-controller  Scaled down replica set nginx-deployment-7b8d4f5c6f to 0
```

输出中需要重点查看的区域包括：

- **Replicas**：此行详细列出了 Pod 状态。它显示了期望数量，以及已更新、总计、可用和不可用的 Pod 数量。在更新过程中，这些数字会不断变动，直到达到期望状态。
- **StrategyType**：确认正在使用的更新策略，默认情况下为 `RollingUpdate`（滚动更新）。
- **Events**：这通常是排除故障最有效的部分。它按时间顺序记录了 Deployment 控制器执行的操作。在上面的示例中，你可以看到控制器在缩减旧 ReplicaSet (`nginx-deployment-7b8d4f5c6f`) 的同时，增加新 ReplicaSet (`nginx-deployment-6c7b5d6f8g`) 的数量，直到转换完成。

> Deployment 在滚动更新期间管理多个 ReplicaSet。图中，一个新的 ReplicaSet 控制着带有更新镜像的 Pod，而旧的 ReplicaSet 随着其 Pod 的终止而缩减副本数。

### 监控实时更新过程

虽然 `kubectl get --watch` 可以显示状态变化，但 `kubectl rollout status` 命令是专门为监控 Deployment 更新直至完成而设计的。该命令会阻塞当前进程并提供实时进度更新，非常适合在自动化脚本或 CI/CD 流水线中使用。

```sh
$ kubectl rollout status deployment/nginx-deployment
Waiting for deployment "nginx-deployment" rollout to finish: 2 out of 3 new replicas have been updated...
Waiting for deployment "nginx-deployment" rollout to finish: 2 out of 3 new replicas have been updated...
Waiting for deployment "nginx-deployment" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "nginx-deployment" rollout to finish: 1 old replicas are pending termination...
deployment "nginx-deployment" successfully rolled out
```

如果更新成功，该命令将以状态码 0 退出。如果更新失败或超时（默认为 10 分钟），它将以非零状态码退出，表示出现了问题。

### 查看 Deployment 历史记录

Deployment 会保留变更的修订历史记录，这使你能够回滚到以前的版本。你可以使用 `kubectl rollout history` 命令查看此历史记录。

```sh
$ kubectl rollout history deployment/nginx-deployment
deployment.apps/nginx-deployment 
REVISION  CHANGE-CAUSE
1         <none>
2         kubectl set image deployment/nginx-deployment nginx=nginx:1.17.1 --record=true
```

每次更新 Deployment 的 Pod 模板（`.spec.template` 字段）时，都会创建一个新的修订版本。

`CHANGE-CAUSE` 列对于了解每个修订版本的创建原因特别有帮助。默认情况下，它的值是 `<none>`。为了自动填充它，你可以在应用变更时将 `kubernetes.io/change-cause` 注解添加到 Deployment 清单中。

例如，你可以先使用 `kubectl apply` 更新清单，然后立即为其添加注解：

```sh
# 应用更新后的 YAML 文件
$ kubectl apply -f nginx-deployment-v3.yaml

# 给 deployment 添加描述性信息的注解
$ kubectl annotate deployment/nginx-deployment kubernetes.io/change-cause="Update to nginx 1.19.0"
```

这种做法使你的部署历史记录更有意义，并在发生问题时简化了确定要回滚到哪个修订版本的过程。

要获取有关特定修订版本的更多详情，请添加 `--revision` 标志：

```sh
$ kubectl rollout history deployment/nginx-deployment --revision=2
```

此命令将向你展示与该特定版本的应用配置相关联的 Pod 模板和其他详细信息。

## 参考资料

- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — Kubernetes Authors (2024)
  解释了Kubernetes Deployment的基本概念、其目的以及如何管理ReplicaSet和Pod以达到预期状态。
- [\`kubectl\` Reference Docs](https://kubernetes.io/docs/reference/kubectl/) — Kubernetes Authors (2024)
  `kubectl`命令的参考资料，涵盖了用于检查Deployment状态和管理回滚的命令。
- [Kubernetes Up & Running: Dive Into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running/9781492046530/) — Brendan Burns, Joe Beda, Kelsey Hightower, and Lachlan Evenson (2019)
  Publisher: O'Reilly Media
  提供在Kubernetes上部署和管理应用程序的实用指南，包括对Deployment和相关检查技术的详细说明。
- [Debugging Deployments](https://kubernetes.io/docs/tasks/debug-application-deployment/) — Kubernetes Authors (2022)
  提供解决Kubernetes Deployment问题的策略和命令，作为本节中检查命令的补充。
