# 执行 Deployment 更新与回滚

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/executing-deployment-updates-rollbacks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在更新运行中的应用程序时，Deployment 对象具有显著优势。Deployment 能够自动执行变更的滚动更新过程，确保从一个应用版本过渡到另一个版本的过程平稳进行，且对用户的影响极小甚至完全没有影响。这种自动化的流程替代了手动停止旧 Pod 并启动新 Pod 的方式，从而降低了操作出错的可能性。

### 滚动更新策略

默认情况下，Deployment 使用 `RollingUpdate`（滚动更新）策略。当你修改 Deployment 规范（specification）中的 Pod 模板（例如更改容器镜像标签）时，Deployment 控制器会启动受控的发布过程。它不会一次性停止所有旧 Pod，而是逐步用新 Pod 替换它们。

具体流程如下：

1. 创建一个带有更新后 Pod 规范的新 `ReplicaSet`。
2. 新 ReplicaSet 开始逐个增加（Scale up）Pod 数量。
3. 与此同时，旧 ReplicaSet 开始减少（Scale down）Pod 数量，终止旧 Pod。
4. 该过程持续进行，直到所有旧 Pod 都被新 Pod 替换，并且新版本的副本数量达到预设值。

由于在更新过程中始终有新旧版本并存来处理流量，这确保了应用在整个更新期间持续可用。

> 此图展示了滚动更新期间的过渡过程。Deployment 控制器管理着两个 ReplicaSet，逐步从旧版本过渡到新版本以保持可用性。

### 执行更新

你可以通过两种主要方式触发更新：使用 `kubectl` 的命令式操作，或通过应用更新后的清单文件进行声明式操作。

#### 使用 kubectl 进行命令式更新

更新应用镜像最快的方法是使用 `kubectl set image` 命令。此命令直接修改集群中运行的 Deployment 对象。

假设你有一个名为 `webapp` 的 Deployment，运行着标签为 `1.21.0` 的 Nginx 镜像。要将其更新为 `1.22.0`，可以运行：

```bash
kubectl set image deployment/webapp nginx=nginx:1.22.0 --record
```

`--record` 标志非常有用，它会将该命令记录在 Deployment 的注解（annotation）中，方便日后查看每个修订版本中所做的具体变更。

#### 使用清单文件进行声明式更新

虽然命令式命令在快速修改时很方便，但在生产环境和 GitOps 工作流中推荐使用声明式方法。你只需修改 Deployment 的 YAML 文件并重新应用即可。

如果原始的 `deployment.yaml` 如下所示：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: nginx
        image: nginx:1.21.0 # 旧镜像版本
        ports:
        - containerPort: 80
```

你需要将 `image` 字段改为 `nginx:1.22.0` 并重新应用清单：

```bash
kubectl apply -f deployment.yaml
```

Kubernetes 会自动计算现有状态与文件中定义的新状态之间的差异（diff），仅在 Pod 模板发生变化时才会触发滚动更新。

### 控制发布速度

你可以通过 Deployment 规范中的两个参数 (parameter)来微调 (fine-tuning)滚动更新过程：`maxUnavailable` 和 `maxSurge`。

- `maxUnavailable`：指定更新期间可以处于不可用状态的 Pod 最大数量。它可以是绝对数值（如 `1`）或所需副本数的百分比（如 `25%`）。
- `maxSurge`：定义可以创建的超过所需副本数量的最大 Pod 数量。这也可以是绝对数值或百分比。

在清单中的定义示例如下：

```yaml
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
```

对于 10 个副本的配置，控制器将确保始终至少有 9 个 Pod 在运行（`10 - 1`）。同时，它也会确保在更新过程中的任何时刻，总 Pod 数量不会超过 11 个（`10 + 1`）。这些设置有助于在更新速度和安全性之间取得平衡。

### 监控发布状态

要查看发布的进展情况，可以使用 `kubectl rollout status` 命令：

```bash
kubectl rollout status deployment/webapp
```

该命令将提供实时反馈，仅在更新完成或失败时才会结束。

```text
Waiting for rollout to finish: 2 of 3 new replicas have been updated...
Waiting for rollout to finish: 2 of 3 new replicas have been updated...
Waiting for rollout to finish: 2 of 3 new replicas have been updated...
Waiting for rollout to finish: 3 of 3 new replicas have been updated...
deployment "webapp" successfully rolled out
```

你还可以使用 `kubectl get deployments` 和 `kubectl get replicasets` 来观察过程中新旧 ReplicaSet 的管理情况。

### 执行回滚

有时新版本可能会引入 Bug。Deployment 提供了一种简单的机制来恢复到之前的版本。

首先，你可以查看 Deployment 的修订历史：

```bash
kubectl rollout history deployment/webapp
```

输出将显示修订版本列表，每一项都对应 Deployment Pod 模板的一个历史版本。

```text
REVISION  CHANGE-CAUSE
1         <none>
2         kubectl set image deployment/webapp nginx=nginx:1.22.0 --record=true
```

`CHANGE-CAUSE` 列由 `--record` 标志或 `kubernetes.io/change-cause` 注解填充，这也体现了养成良好记录习惯的原因。

要回滚到上一个版本（在本例中为修订版本 1），可以使用 `undo` 命令：

```bash
kubectl rollout undo deployment/webapp
```

这将触发另一次滚动更新，但这次是将 Pod 模板还原为上一个修订版本的配置。

如果你需要回滚到特定的修订版本，可以使用 `--to-revision` 标志指定：

```bash
kubectl rollout undo deployment/webapp --to-revision=1
```

这种回滚能力提供了一层安全保障，让你能够快速从有问题的部署中恢复，而无需手动干预。通过管理从部署、更新到回滚的完整应用生命周期，Deployment 为在 Kubernetes 上运行无状态应用提供了一种可靠且自动化的方式。

## 参考资料

- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — Kubernetes Authors (2024)
  这份官方指南详细介绍了Kubernetes Deployment对象、其滚动更新策略以及它如何管理应用程序的生命周期，包括更新和回滚。
- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications
  这是一本全面的书籍，提供了Kubernetes核心概念的实践解释和示例，包括对Deployment、滚动更新和回滚的讲解。
- [kubectl rollout](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#rollout) — Kubernetes Authors (2020)
  这是`kubectl`管理回滚的官方命令参考，包括用于查看状态、历史记录和撤销部署的命令，直接支持了讨论中的实践操作。

---

[上一节](04-%E5%AE%9A%E4%B9%89%20Deployment%20%E6%B8%85%E5%8D%95.md) · [下一节](06-%E6%A3%80%E6%9F%A5%20Deployment%20%E7%8A%B6%E6%80%81.md)
