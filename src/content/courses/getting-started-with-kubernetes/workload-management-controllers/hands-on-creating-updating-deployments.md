---
course: "getting-started-with-kubernetes"
chapter: "workload-management-controllers"
lesson: "hands-on-creating-updating-deployments"
sourceId: 7315
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/hands-on-creating-updating-deployments"
title: "动手实践：创建与更新 Deployment"
description: "动手练习创建 Deployment、进行扩缩容、执行到新应用版本的滚动更新以及执行回滚操作。"
order: 7
plots: []
sourceHash: "94f36da35f7aea73be1d8d348c474eea9fe3f837776bfdf9ad33dd0d03d3baac"
sourceCorrections: []
---

使用 `kubectl`，可以对 Deployment 进行创建、查看、扩缩容、更新和回滚操作。这些操作展示了 Deployment 如何实现应用程序生命周期的自动化。本节将部署一个简单的 NGINX Web 服务器。

### 前提条件

在开始之前，请确保你有一个正在运行的 Kubernetes 集群，并且已配置好 `kubectl` 与其交互。如果你按照第 1 章的设置指南进行操作，那么你的本地 Minikube 或 Kind 集群已经为本次练习准备就绪。

### 第 1 步：定义 Deployment

首先，我们创建一个 YAML 清单文件来描述期望状态。这种声明式方法是有效管理 Kubernetes 资源的核心。创建一个名为 `nginx-deployment.yaml` 的文件，并添加以下内容：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21.6
        ports:
        - containerPort: 80
```

让我们简要查看一下这个清单的结构：

- `apiVersion: apps/v1`：指定 Deployment 对象的 API 版本。
- `kind: Deployment`：将资源类型定义为 Deployment。
- `metadata.name`：将我们的 Deployment 命名为 `nginx-deployment`。
- `spec.replicas: 3`：指示控制器为该应用维持正好三个运行中的 Pod。
- `spec.selector`：告知 Deployment 管理哪些 Pod。`matchLabels` 字段必须与 Pod 模板（`spec.template.metadata.labels`）中定义的标签匹配。Deployment 正是通过这种关联来找到它所属的 Pod。
- `spec.template`：这是 Deployment 将要创建的 Pod 的蓝图。它包含标准的 Pod `metadata`（带有标签）和 `spec`（带有容器定义）。
- `spec.template.spec.containers`：这里我们定义了一个使用 `nginx:1.21.6` 镜像的容器。

### 第 2 步：创建并查看 Deployment

清单文件准备好后，使用 `kubectl apply` 将其应用到集群。此命令将清单发送到 Kubernetes API 服务器，随后服务器开始创建资源。

```bash
kubectl apply -f nginx-deployment.yaml
```

你应该会看到输出 `deployment.apps/nginx-deployment created`。Kubernetes 现在开始工作，使集群的当前状态与文件中定义的期望状态保持一致。Deployment 控制器会创建一个 ReplicaSet，而 ReplicaSet 又会根据你的模板创建三个 Pod。

让我们验证一下创建的资源。

首先，检查 Deployment 本身的状态：

```bash
kubectl get deployments
```

输出应显示期望（desired）、当前（current）、最新（up-to-date）和可用（available）的副本数均为 3。

```
NAME               READY   UP-TO-DATE   AVAILABLE   AGE
nginx-deployment   3/3     3            3           15s
```

接着，查看由 Deployment 创建的 ReplicaSet。你会注意到它的名称是由 Deployment 名称加上一个唯一的哈希值生成的。

```bash
kubectl get replicasets
```

```
NAME                          DESIRED   CURRENT   READY   AGE
nginx-deployment-65c65b4f6d   3         3         3       30s
```

最后，列出 Pod 以确认有三个实例正在运行：

```bash
kubectl get pods
```

```
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-65c65b4f6d-7vjpl   1/1     Running   0          45s
nginx-deployment-65c65b4f6d-8xqz8   1/1     Running   0          45s
nginx-deployment-65c65b4f6d-kbnm4   1/1     Running   0          45s
```

### 第 3 步：扩缩容应用程序

你的应用流量正在增加，需要将实例从三个增加到五个。Deployment 控制器让这一切变得简单。你可以将 `nginx-deployment.yaml` 文件中的 `replicas` 字段更新为 `5` 并重新应用。

```yaml
# 在 nginx-deployment.yaml 中
...
spec:
  replicas: 5 # 从 3 修改为 5
...
```

应用更新后的清单：

```bash
kubectl apply -f nginx-deployment.yaml
```

Kubernetes 会检测到变化并调和状态。Deployment 控制器更新 ReplicaSet 的期望数量，随后 ReplicaSet 控制器创建两个新 Pod。

再次列出 Pod 来验证扩容操作：

```bash
kubectl get pods
```

```
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-65c65b4f6d-7vjpl   1/1     Running   0          3m
nginx-deployment-65c65b4f6d-8xqz8   1/1     Running   0          3m
nginx-deployment-65c65b4f6d-kbnm4   1/1     Running   0          3m
nginx-deployment-65c65b4f6d-pxr9t   1/1     Running   0          30s
nginx-deployment-65c65b4f6d-q7z2l   1/1     Running   0          30s
```

你现在有五个正在运行的 Pod。缩容也同样容易，只需减少 `replicas` 数量并重新应用清单即可。

> **命令式与声明式扩缩容**
> 你也可以使用命令式命令（如 `kubectl scale deployment nginx-deployment --replicas=5`）来扩缩 Deployment。虽然这对于立即调整很方便，但它会导致运行中的配置与你的清单文件之间出现不一致。通常推荐使用声明式的 `kubectl apply` 方法，因为它能让版本控制的清单文件始终作为集群状态的单一事实来源。

### 第 4 步：执行滚动更新

Deployment 擅长管理零停机时间的应用程序更新。让我们将 NGINX 应用程序从版本 `1.21.6` 更新到 `1.23.3`。

修改 `nginx-deployment.yaml` 中的 `image` 字段：

```yaml
# 在 nginx-deployment.yaml 中
...
    spec:
      containers:
      - name: nginx
        image: nginx:1.23.3 # 从 nginx:1.21.6 修改而来
...
```

应用更改：

```bash
kubectl apply -f nginx-deployment.yaml
```

当你应用此更改时，Deployment 控制器会启动**滚动更新**。它为新版本（`1.23.3`）创建一个新的 ReplicaSet，并逐渐增加其副本数，同时减少旧版本（`1.21.6`）ReplicaSet 的副本数。这确保了应用程序在整个更新过程中始终可用。

> 由 Deployment 管理的滚动更新过程。控制器从旧的 ReplicaSet 逐步过渡到新的 ReplicaSet，确保服务可用性。

你可以实时监控滚动更新的状态：

```bash
kubectl rollout status deployment/nginx-deployment
```

更新完成后，该命令将退出。在更新过程中，你可以打开另一个终端并运行 `kubectl get pods`，观察新 Pod 的创建和旧 Pod 的终止。

### 第 5 步：回滚有问题的更新

假设新版本 `1.23.3` 存在严重缺陷。你需要立即恢复到之前的稳定版本。Deployment 维护着修订历史记录，使得回滚非常直接。

首先，查看 Deployment 的修订历史：

```bash
kubectl rollout history deployment/nginx-deployment
```

```
REVISION  CHANGE-CAUSE
1         <none>
2         <none>
```

你可以看到两个修订版本。修订版 1 对应于最初使用镜像 `1.21.6` 创建的状态，修订版 2 对应于更新到 `1.23.3` 的状态。

要回滚到上一个版本（修订版 1），请使用 `rollout undo` 命令：

```bash
kubectl rollout undo deployment/nginx-deployment
```

此命令指示 Deployment 恢复到之前的配置。控制器现在将执行另一次滚动更新，这次是缩小新 ReplicaSet (`1.23.3`) 的规模，并扩大旧 ReplicaSet (`1.21.6`) 的规模。

你可以通过查看 Deployment 详情并检查镜像标签来验证回滚情况：

```bash
kubectl describe deployment nginx-deployment | grep Image
```

输出将显示镜像已恢复为 `nginx:1.21.6`。

### 第 6 步：清理资源

完成后，你可以通过一个命令删除 Deployment 及其创建的所有资源（ReplicaSet 和 Pod）：

```bash
kubectl delete -f nginx-deployment.yaml
```

此命令会移除 `nginx-deployment`，并指示 Kubernetes 终止其关联的 ReplicaSet 和 Pod，从而清理集群。

## 参考资料

- [Kubernetes: Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and-running-2nd/9781492046530/) — Brendan Burns, Joe Beda, Kelsey Hightower (2019)
  Publisher: O'Reilly Media; Pages: 275
  提供了Kubernetes架构和组件的深入说明，包括从基础到高级的部署、副本集和`kubectl`操作。
- [Overview of kubectl | Kubernetes](https://kubernetes.io/docs/reference/kubectl/overview/) — Kubernetes Authors (2024)
  提供了`kubectl`命令行工具的指南，涵盖其语法、常用命令以及如何与Kubernetes集群交互。
