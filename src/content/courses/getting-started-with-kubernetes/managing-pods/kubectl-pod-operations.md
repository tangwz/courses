---
course: "getting-started-with-kubernetes"
chapter: "managing-pods"
lesson: "kubectl-pod-operations"
sourceId: 7306
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods/kubectl-pod-operations"
title: "使用 kubectl 进行 Pod 操作"
description: "在集群中创建、检查、删除和排查 Pod 故障时常用的 kubectl 命令参考。"
order: 5
plots: []
sourceHash: "8e28af28ac3c83caba5b9d9a60ca5c5acce9328d0769dfaccbb40693efb49afd"
sourceCorrections: []
---

管理 Kubernetes Pod 包括在 YAML 配置清单中定义它们，然后将其提交给 Kubernetes API 服务器以控制它们的生命周期。用于此交互的主要工具是 `kubectl`，它是 Kubernetes 官方命令行界面。它允许你执行每项集群操作，从部署应用到检查它们的运行状态。以下是用于创建、检查、调试和删除 Pod 的常用 `kubectl` 命令。

### 通过配置清单创建 Pod

创建 Pod 最常见的方法是应用 YAML 配置清单文件。`kubectl apply` 命令指示 Kubernetes 创建或更新资源，使其与文件中定义的状态相匹配。这种声明式方法是 Kubernetes 管理的核心，因为它允许系统确定实现所需状态所需的行动。

假设你有一个保存为 `simple-pod.yaml` 的 Pod 配置清单：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-server
  labels:
    app: nginx
spec:
  containers:
  - name: nginx-container
    image: nginx:1.23
    ports:
    - containerPort: 80
```

要创建此 Pod，请在文件所在的目录中从终端运行 `apply` 命令：

```bash
kubectl apply -f simple-pod.yaml
```

如果成功，API 服务器将通过简单的确认信息来确认请求：

```
pod/web-server created
```

此时，Kubernetes 已接受该配置清单，并将开始调度和启动 Pod 的流程。

> 此图显示了创建 Pod 的高级工作流。用户在 YAML 配置清单中记录的意图通过 `kubectl` 传递给 API 服务器。随后，控制平面组件协作，在合适的宿主机节点上调度并运行 Pod。

### 检查 Pod 状态

Pod 创建后，你需要检查其状态，查找其 IP 地址并检查其配置。

#### 列出 Pod

`kubectl get pods` 命令提供当前命名空间中一个或多个 Pod 的状态摘要。

```bash
kubectl get pods
```

输出是一个包含每个 Pod 基本信息的表格：

```
NAME         READY   STATUS    RESTARTS   AGE
web-server   1/1     Running   0          2m15s
```

- **NAME:** Pod 的名称，如其元数据中所定义。
- **READY:** 显示 Pod 中处于“就绪”状态的容器数量占容器总数的比例。`1/1` 表示单个容器正在运行并通过了就绪探针（如果已配置）。
- **STATUS:** Pod 在其生命周期中的当前阶段，例如 `Pending`（等待中）、`Running`（运行中）、`Succeeded`（成功）或 `Failed`（失败）。
- **RESTARTS:** Pod 中容器重启的次数。此数值较高通常表示存在问题。
- **AGE:** 自 Pod 创建以来经过的时间。

要查看更多信息（例如 Pod 的 IP 地址及其被调度到的节点），请使用 `-o wide` 标志。

```bash
kubectl get pods -o wide
```

```
NAME         READY   STATUS    RESTARTS   AGE     IP          NODE       NOMINATED NODE   READINESS GATES
web-server   1/1     Running   0          3m30s   10.244.1.4   minikube   <none>           <none>
```

#### 查看 Pod 详情

要全面了解 Pod 的配置和运行时状态，请使用 `kubectl describe pod`。此命令提供详细的、人类可读的输出，对故障排除非常有用。

```bash
kubectl describe pod web-server
```

输出内容详尽但结构清晰。主要部分包括：

- **标签（Labels）和注解（Annotations）：** 附加到 Pod 的所有元数据。
- **状态（Status）：** 当前生命周期阶段。
- **IP：** 分配给 Pod 的 IP 地址。
- **容器（Containers）：** 每个容器的详细信息，包括镜像 ID、端口和资源限制。
- **事件（Events）：** 与 Pod 相关的事件按时间顺序排列的日志，例如被调度、拉取镜像和启动容器。当 Pod 运行不正常时，这通常是首先要查看的地方。

以下是一个健康 Pod 的 `Events` 部分片段：

```
Events:
  Type    Reason     Age    From               Message
  ----    ------     ----   ----               -------
  Normal  Scheduled  5m10s  default-scheduler  Successfully assigned default/web-server to minikube
  Normal  Pulling    5m9s   kubelet            Pulling image "nginx:1.23"
  Normal  Pulled     5m5s   kubelet            Successfully pulled image "nginx:1.23" in 4.1s
  Normal  Created    5m5s   kubelet            Created container nginx-container
  Normal  Started    5m5s   kubelet            Started container nginx-container
```

### 获取 Pod 日志

为了排查应用行为故障，你需要访问其容器生成的日志。`kubectl logs` 命令会流式传输容器的标准输出和标准错误。

查看 `web-server` Pod 的当前日志：

```bash
kubectl logs web-server
```

如果 Pod 包含多个容器，你必须使用 `-c` 或 `--container` 标志指定要查看哪个容器的日志：

```bash
kubectl logs my-multi-container-pod -c sidecar-container
```

要实时跟踪日志流（类似于 Linux 中的 `tail -f`），请使用 `-f` 标志：

```bash
kubectl logs -f web-server
```

### 在 Pod 内部执行命令

有时，仅查看日志是不够的。为了进行更细致的调试，你可能需要直接在运行中的容器内运行命令。`kubectl exec` 提供了这种能力。

要在 `web-server` 容器内获取交互式 shell 会话，请使用 `-it` 标志（`-i` 代表标准输入，`-t` 代表终端）：

```bash
kubectl exec -it web-server -- /bin/bash
```

`--` 将 `kubectl` 命令及其参数 (parameter)与你想在容器内运行的命令分隔开。进入容器后，你可以检查网络连接、查看文件系统或检查运行中的进程。

你也可以运行非交互式命令。例如，列出 Nginx 网页根目录中的文件：

```bash
kubectl exec web-server -- ls -l /usr/share/nginx/html
```

### 删除 Pod

最后，要从集群中移除 Pod，可以使用 `kubectl delete`。你可以通过名称删除 Pod，也可以通过重新应用创建它的文件来删除。

按名称删除 `web-server` Pod：

```bash
kubectl delete pod web-server
```

或者，使用配置清单文件也是一种常见做法：

```bash
kubectl delete -f simple-pod.yaml
```

这两个命令都会终止 Pod。Kubernetes 将通知容器正常关闭，如果容器在特定的宽限期内没有响应，则会强制将其终止。

这些命令构成了与 Pod 交互的基础。随着学习的深入，你会发现类似的命令结构也用于管理其他 Kubernetes 资源，如 Deployment 和 Service。

## 参考资料

- [\`kubectl\` Overview](https://kubernetes.io/docs/reference/kubectl/) — Kubernetes Authors (2024)
  提供 `kubectl` 命令及其用法的详细信息。
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) — Kubernetes Authors (2024)
  描述 Kubernetes 中运行容器的基础资源。
- [Kubernetes: Up & Running: Dive into the Future of Infrastructure](https://books.google.com/books/about/Kubernetes_Up_and_Running.html?id=z17uEAAAQBAJ) — Brendan Burns, Joe Beda, Kelsey Hightower, and Lachlan Evenson (2022)
  Publisher: O'Reilly Media; Pages: 326
  一本涵盖 Kubernetes 基础知识的著作，包括使用 `kubectl` 进行资源管理。
