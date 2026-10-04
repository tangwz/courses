---
course: "getting-started-with-kubernetes"
chapter: "managing-pods"
lesson: "practice-deploying-inspecting-pods"
sourceId: 7308
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods/practice-deploying-inspecting-pods"
title: "练习：部署与检查 Pod"
description: "一个实操练习课程，你将编写 Pod 清单，将其部署到集群，并使用 kubectl 检查其状态和日志。"
order: 7
plots: []
sourceHash: "f79a29d1760bacfd370de277af0a0c47fb97562982fc561bc7c08342d7688c25"
sourceCorrections: []
---

你将在本地 Kubernetes 集群中创建、部署、检查并管理一个 Pod，应用有效的 Pod 管理方法。你将直接使用 `kubectl` 和 YAML 清单，以巩固对 Pod 生命周期及其运维管理的理解。

### 目标

完成此练习后，你将能够：

- 为单容器应用编写基础的 Pod 清单。
- 使用 `kubectl` 将 Pod 部署到集群。
- 检查 Pod 的状态、事件和日志。
- 从本地机器访问 Pod 内运行的应用程序。
- 修改清单以包含健康检查探针。
- 清理创建的资源。

在开始之前，请确保你有一个正在运行的本地 Kubernetes 集群，并且已按照第 1 章所述配置好 `kubectl` 以便与其交互。

### 第 1 步：编写并部署 Pod 清单

我们的第一个任务是创建一个运行简单 NGINX Web 服务器的 Pod。这需要定义一个指定预期状态的清单文件。

创建一个名为 `nginx-pod.yaml` 的新文件，并添加以下内容：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-server
  labels:
    app: nginx
spec:
  containers:
  - name: web-server
    image: nginx:1.25
    ports:
    - containerPort: 80
```

让我们简要查看此清单中的字段：

- `apiVersion: v1`：指定 Pod 对象的 API 版本，属于核心 API 组。
- `kind: Pod`：声明我们正在定义一个 Pod 对象。
- `metadata`：包含有助于唯一标识对象的数据，包括其 `name`（名称）和描述性的 `labels`（标签）。
- `spec`：定义 Pod 的预期状态。
- `containers`：要在 Pod 中运行的一个或多个容器的列表。
  - `name`：Pod 内容器的名称。
  - `image`：要从镜像仓库拉取的容器镜像。
  - `ports`：要从容器暴露的网络端口列表。`containerPort: 80` 告知 Kubernetes 该 NGINX 容器正在监听 80 端口。

现在，使用 `kubectl` 将此清单应用到你的集群：

```sh
kubectl apply -f nginx-pod.yaml
```

你应该会看到一条确认消息：`pod/nginx-server created`。此命令指示 Kubernetes 读取清单并创建对象，努力使集群状态与你声明的状态相匹配。

### 第 2 步：检查 Pod

应用清单后，Kubernetes 控制平面会开始调度并运行 Pod。我们可以使用几个 `kubectl` 命令来观察这个过程。

首先，检查 Pod 的状态。在拉取镜像时，状态可能需要一点时间从 `Pending`（待定）或 `ContainerCreating`（正在创建容器）转变为 `Running`（正在运行）。

```sh
kubectl get pod nginx-server
```

输出结果将类似于：

```
NAME           READY   STATUS    RESTARTS   AGE
nginx-server   1/1     Running   0          30s
```

`STATUS` 列显示 Pod 正在运行，`READY` 列表示 Pod 中的 `1/1` 个容器已准备好处理流量。

要查看更详细的信息，请使用 `describe` 命令。这是排查故障的利器。

```sh
kubectl describe pod nginx-server
```

输出内容较多，但提供了丰富的信息。请关注以下部分：

- **IP：** 在集群网络内分配给 Pod 的内部 IP 地址。
- **Containers：** 有关容器状态、镜像 ID 和端口映射的详细信息。
- **Events：** 调度器和 kubelet 针对此 Pod 采取的操作的时间线记录。如果 Pod 启动失败，原因通常可以在这里找到。

```
...
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  1m    default-scheduler  Successfully assigned default/nginx-server to minikube
  Normal  Pulling    1m    kubelet            Pulling image "nginx:1.25"
  Normal  Pulled     50s   kubelet            Successfully pulled image "nginx:1.25" in 10s
  Normal  Created    50s   kubelet            Created container web-server
  Normal  Started    50s   kubelet            Started container web-server
```

### 第 3 步：与运行中的应用程序交互

我们的 NGINX 容器正在 Pod 内部运行，但只能在集群的专用网络内访问。为了从本地机器与其交互，我们可以使用 `kubectl port-forward`。此命令会在你的本地机器和 Pod 之间创建一个安全隧道。

打开一个新的终端窗口并运行以下命令：

```sh
kubectl port-forward pod/nginx-server 8080:80
```

此命令将流量从本地端口 `8080` 转发到 Pod 的容器端口 `80`。只要隧道保持开启，该命令就会一直处于活动状态。

现在，你可以访问 NGINX 欢迎页面。使用浏览器访问 `http://localhost:8080` 或在另一个终端中使用 `curl`：

```sh
curl http://localhost:8080
```

你应该能看到标准的 NGINX HTML 响应，确认你的应用正在运行且可访问。

你还可以查看容器生成的日志。回到第一个终端（未运行端口转发的那个）并执行：

```sh
kubectl logs nginx-server
```

这将显示 NGINX 访问日志，包括你刚才用 `curl` 发出的请求。要在容器内使用交互式 shell，请使用 `exec`：

```sh
kubectl exec -it nginx-server -- /bin/bash
```

这会为你提供 `web-server` 容器内的 root shell。在这里，你可以检查文件系统和正在运行的进程。输入 `exit` 返回本地 shell。

### 第 4 步：添加存活和就绪探针

运行中的应用程序并不总是健康的。让我们修改清单以包含健康探针，让 Kubernetes 能够自动管理 Pod 的健康状态。

使用以下 `livenessProbe`（存活探针）和 `readinessProbe`（就绪探针）更新 `nginx-pod.yaml`：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-server
  labels:
    app: nginx
spec:
  containers:
  - name: web-server
    image: nginx:1.25
    ports:
    - containerPort: 80
    readinessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 10
    livenessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 15
      periodSeconds: 20
```

以下是我们添加的内容：

- **readinessProbe：** kubelet 将向端口 80 上的 `/` 发送 HTTP GET 请求。如果收到成功代码，则 Pod 被标记 (token)为 `Ready`（就绪）。在就绪探针成功之前，Kubernetes 不会将流量发送到该 Pod。
- **livenessProbe：** 如果此探针失败，kubelet 将杀死容器并根据 Pod 的重启策略将其重启。这有助于从死锁的应用中恢复。

保存文件并重新应用。Kubernetes 能够使用新配置更新现有对象。

```sh
kubectl apply -f nginx-pod.yaml
```

你会看到 `pod/nginx-server configured`。再次描述 Pod 以查看新的探针定义：

```sh
kubectl describe pod nginx-server
```

注意容器描述中新增的 `Liveness` 和 `Readiness` 部分。Pod 的状态也会在 `kubectl get pod` 中反映就绪状态。

### 第 5 步：清理

为了结束本次练习，我们将从集群中删除该 Pod。你可以按名称进行命令式删除，也可以使用创建时使用的文件进行声明式删除。通常首选使用文件，因为这能确保你删除的内容与定义的内容完全一致。

```sh
kubectl delete -f nginx-pod.yaml
```

输出 `pod "nginx-server" deleted` 确认资源已被移除。你可以通过运行 `kubectl get pods` 来验证。到这里，我们关于创建和检查基础 Kubernetes 对象的实操练习就完成了。

## 参考资料

- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) — Kubernetes Authors (2024)
  Kubernetes Pod 的基础指南，涵盖其作用、生命周期和基本配置。
- [kubectl Overview](https://kubernetes.io/docs/reference/kubectl/overview/) — Kubernetes Authors (2024)
  `kubectl` 的官方参考资料，详细介绍了其命令以及如何与 Kubernetes 集群交互。
- [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/) — Kubernetes Authors (2024)
  关于如何在 Kubernetes Pod 中的容器内实现和配置健康探针的实用指南。
