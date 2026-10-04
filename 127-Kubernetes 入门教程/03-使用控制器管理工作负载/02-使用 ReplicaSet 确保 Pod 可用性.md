# 使用 ReplicaSet 确保 Pod 可用性

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/replicasets-for-pod-availability)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 Kubernetes 中，确保 Pod 的可用性和弹性是通过一个名为 **ReplicaSet** 的专用控制器实现的。该控制器的主要任务是确保在任何给定时间都有指定数量的 Pod 副本在运行，从而维持一组稳定的 Pod。这种能力为应用程序提供了自愈和伸缩的基础层。

ReplicaSet 的工作方式是持续监控集群状态。如果它管理的 Pod 数量低于期望值（可能是由于节点故障或误删除导致的），ReplicaSet 的控制循环会自动创建新的 Pod 来弥补。相反，如果 Pod 数量过多，它会终止多余的 Pod 以匹配期望状态。

### ReplicaSet 清单的结构

与其他 Kubernetes 对象一样，ReplicaSet 也是使用 YAML 清单定义的。在其 `spec` 中，定义由三个主要部分组成：`replicas`（副本数量）、`selector`（用于识别其管理的 Pod 的选择器）以及 `template`（用于在需要时创建新 Pod 的模板）。

让我们看一个典型的 ReplicaSet 清单：

```yaml
# myapp-replicaset.yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: myapp-rs
  labels:
    app: myapp
spec:
  # 1. 期望的 Pod 数量
  replicas: 3

  # 2. 用于识别受管 Pod 的选择器
  selector:
    matchLabels:
      app: myapp-pod

  # 3. 创建新 Pod 的模板
  template:
    metadata:
      labels:
        app: myapp-pod
    spec:
      containers:
      - name: nginx-container
        image: nginx:1.21.6
```

1. **`replicas`**：这个简单的整数值声明了期望状态。在这里，ReplicaSet 被配置为精确维持三个 Pod。你可以随时更改此值来对应用程序进行扩缩容。
2. **`selector`**：该字段定义了 ReplicaSet 如何查找需要管理的 Pod。它使用标签来建立这种关联。在本例中，ReplicaSet 会监控其命名空间中任何带有 `app: myapp-pod` 标签的 Pod。这种解耦方式非常实用；只要标签匹配，ReplicaSet 甚至可以管理那些并非由它创建的 Pod。
3. **`template`**：此部分是创建新 Pod 的蓝图。当 ReplicaSet 需要创建 Pod 以满足 `replicas` 数量时，就会使用此模板。模板包含自己的 `metadata`（包括标签）和 `spec`（包括容器定义），就像独立的 Pod 清单一样。

有一项强制要求：`spec.template.metadata.labels` 中定义的标签必须与 `spec.selector.matchLabels` 中的标签匹配。如果不匹配，Kubernetes API 将拒绝该 ReplicaSet。这可以防止控制器创建了 Pod 却在随后无法找到并管理它们的情况。

### 调解循环的工作原理

ReplicaSet 控制器会持续运行一个称为“调解循环”的过程。该循环将期望状态（来自 `replicas` 字段）与实际状态（符合 `selector` 条件的运行中 Pod 数量）进行比较，并采取措施纠正任何偏差。

> ReplicaSet 的调解循环确保活动 Pod 的数量始终与期望的副本数保持一致。

### 创建并查看 ReplicaSet

你可以使用 `kubectl apply` 命令通过清单文件创建 ReplicaSet。

```bash
$ kubectl apply -f myapp-replicaset.yaml
replicaset.apps/myapp-rs created
```

创建后，你可以使用 `kubectl get rs` 查看其状态。

```bash
$ kubectl get rs
NAME       DESIRED   CURRENT   READY   AGE
myapp-rs   3         3         3       12s
```

输出显示“期望（DESIRED）”状态为 3，“当前（CURRENT）”Pod 数量为 3，且 3 个都处于“就绪（READY）”状态，可以处理流量。

要查看由此 ReplicaSet 管理的 Pod，可以使用选择器的标签：

```bash
$ kubectl get pods -l app=myapp-pod
NAME             READY   STATUS    RESTARTS   AGE
myapp-rs-5x2kz   1/1     Running   0          45s
myapp-rs-h7j8p   1/1     Running   0          45s
myapp-rs-w9n4f   1/1     Running   0          45s
```

请注意，Pod 名称是通过在 ReplicaSet 名称后添加随机哈希值生成的，以确保唯一性。

### 演示自愈功能

ReplicaSet 的真正价值在于它能从故障中自动恢复。为了观察这一现象，我们可以手动删除其中一个 Pod：

```bash
# 获取要删除的 Pod 名称
$ POD_NAME=$(kubectl get pods -l app=myapp-pod -o jsonpath='{.items[0].metadata.name}')

# 删除 Pod
$ kubectl delete pod $POD_NAME
pod "myapp-rs-5x2kz" deleted
```

现在，立即再次检查 Pod。你可能需要快速运行命令才能看到整个周期。

```bash
$ kubectl get pods -l app=myapp-pod
NAME             READY   STATUS              RESTARTS   AGE
myapp-rs-h7j8p   1/1     Running             0          2m10s
myapp-rs-w9n4f   1/1     Running             0          2m10s
myapp-rs-z1b3x   0/1     ContainerCreating   0          2s
```

ReplicaSet 检测到只有两个 Pod 在运行，未达到期望的三个。它立即使用其模板创建了一个新 Pod (`myapp-rs-z1b3x`) 来替换被终止的那个。这种自愈能力是在 Kubernetes 上运行高可用应用程序的基础。

### ReplicaSet 在现代工作负载中的角色

虽然 ReplicaSet 在维护稳定副本数量方面非常有效，但在现代 Kubernetes 工作流中，它们被视为较低级别的构建块。它们没有内置的应用程序更新机制，例如在不中断服务的情况下更改容器镜像版本。

因此，在日常工作中你很少会直接定义 ReplicaSet。相反，你会使用一个名为 **Deployment** 的更高级别控制器，它会代表你管理 ReplicaSet，从而实现复杂的更新和回滚操作。我们将在下一节中介绍 Deployment。

## 参考资料

- [ReplicaSet](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/) — Kubernetes Authors (2024)
  官方文档，描述了Kubernetes ReplicaSet 的配置及其在维持所需 Pod 数量方面的作用。
- [Controllers](https://kubernetes.io/docs/concepts/architecture/controller/) — Kubernetes Authors (2024)
  提供了Kubernetes 控制器的概述，解释了它们在集群中管理资源和维持所需状态的作用。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure, 3rd Edition](https://www.oreilly.com/library/view/kubernetes-up-running/9781098110192/) — Brendan Burns, Joe Beda, Kelsey Hightower, and Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本广受好评的著作，提供了关于Kubernetes 架构、基本对象和常见操作模式的实用指南。
- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — Kubernetes Authors (2024)
  Kubernetes Deployment 的官方文档，Deployment 是用于声明式更新和回滚的高级控制器，负责管理 ReplicaSet。

---

[上一节](01-Kubernetes%20%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20Deployment%20%E8%BF%9B%E8%A1%8C%E5%BA%94%E7%94%A8%E5%8F%91%E5%B8%83.md)
