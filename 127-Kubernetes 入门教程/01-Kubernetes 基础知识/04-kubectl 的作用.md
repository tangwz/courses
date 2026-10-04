# kubectl 的作用

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/role-of-kubectl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要与控制平面的组件交互并管理您的应用程序，您需要一个客户端。虽然您可以直接向 API Server 构建原始 HTTP 请求，但这样做既繁琐又容易出错。与 Kubernetes 集群通信的标准且最有效的方式是通过其命令行界面（CLI），即 `kubectl`。

可以将 `kubectl` 视为任何 Kubernetes 集群的通用遥控器。它是一个功能强大的客户端，可将您的命令转换为规范的 API 请求，发送到 API Server 进行处理，并为您格式化响应结果。无论您是检查节点的健康状况、部署新应用程序，还是调试正在运行的容器，`kubectl` 都是您会用到的工具。

### kubectl 通信流程

您使用 `kubectl` 执行的每个命令都会启动一个标准的客户端-服务器交互。当您运行类似 `kubectl get nodes` 的命令时，会发生以下一系列事件：

1. **配置检查：** `kubectl` 首先从 `kubeconfig` 文件中读取配置，以确定要联系哪个集群的 API Server，以及使用哪些凭据进行身份验证。
2. **API 请求：** 它将您的命令转换为 RESTful API 调用。对于 `kubectl get nodes`，这会变成一个向 API Server 的 `/api/v1/nodes` 端点发出的 HTTP GET 请求。
3. **API Server 处理：** API Server 接收请求。它会对用户进行身份验证和授权，验证请求，然后从集群的状态存储库 etcd 中检索请求的信息。
4. **响应：** API Server 返回响应，通常为 JSON 格式，其中包含节点列表及其状态。
5. **格式化：** `kubectl` 接收此 JSON 数据并将其格式化为易于阅读的表格，然后打印到您的终端。

这种交互是 Kubernetes 内部所有操作的基础。

> 命令行交互流程，其中 `kubectl` 充当用户与 Kubernetes API Server 之间的中介。

### 使用 kubeconfig 配置访问

为了让 `kubectl` 连接到集群，它依赖于一个配置文件。该文件通常位于 `~/.kube/config`，包含查找一个或多个 Kubernetes 集群并进行身份验证所需的详细信息。它采用 YAML 结构，由三个主要部分组成：

- **clusters（集群）：** 定义集群列表，每个集群包含名称、API Server URL 以及信任服务器身份所需的证书颁发机构数据。
- **users（用户）：** 包含客户端的身份验证凭据。用户可以通过客户端证书、不记名令牌（bearer token）或其他身份验证方法来定义。
- **contexts（上下文 (context)）：** 将用户与集群绑定。上下文定义了一个工作环境，允许您指定在与特定集群通信时使用哪些用户凭据。您还可以为该上下文指定默认的命名空间。

`kubectl` 使用 `current-context` 的值来确定当前处于活动状态的上下文。这种设计使得从单个工作站管理对开发、测试和生产等多个环境的访问变得非常直接。

您可以通过运行以下命令查看当前配置：

```sh
kubectl config view
```

此命令显示合并后的 `kubeconfig` 设置。要在不同的上下文之间切换，请使用命令 `kubectl config use-context <context-name>`。

### kubectl 命令的结构

大多数 `kubectl` 命令遵循一致的结构，使其具有可预测性且易于学习：

```
kubectl [command] [TYPE] [NAME] [flags]
```

让我们分解每个部分：

- **`[command]`**：指定您要执行的操作。这些是动词，例如 `get`（列出资源）、`describe`（显示详细信息）、`apply`（通过文件创建或更新）、`delete`（移除资源）和 `exec`（在容器中执行命令）。
- **`[TYPE]`**：您要操作的资源类型。例如 `pod`、`deployment`、`service` 和 `node`。资源类型通常有简短的别名，例如 `po` 代表 pod，`svc` 代表 service。
- **`[NAME]`**：特定资源的名称。如果在执行 `get` 时省略名称，`kubectl` 将列出指定类型的所有资源。对于 `describe` 或 `delete` 等命令，通常需要指定名称。
- **`[flags]`**：修改命令行为的可选参数 (parameter)。例如，`-n <namespace>` 指定命名空间，`-o yaml` 将输出格式更改为 YAML，`--all-namespaces`（或 `-A`）则针对所有命名空间中的资源。

### 日常使用的基础命令

虽然 `kubectl` 有几十个命令，但在管理和检查应用程序时，经常会用到以下几个。

**检查集群资源：**

- 列出当前命名空间中的所有 pod：

  ```sh
  kubectl get pods
  ```
- 获取有关特定 pod 的详细信息，包括事件、标签和容器状态：

  ```sh
  kubectl describe pod my-app-pod-12345
  ```
- 查看 pod 中正在运行的容器的日志：

  ```sh
  kubectl logs my-app-pod-12345
  ```
- 在正在运行的容器中打开交互式 shell 以进行调试：

  ```sh
  kubectl exec -it my-app-pod-12345 -- /bin/sh
  ```

**管理资源：**

- 创建或更新资源的推荐方法是声明式的，即使用 YAML 清单文件。这种方法是 Kubernetes 操作的核心，将在下一节中讲解。

  ```sh
  kubectl apply -f my-app-deployment.yaml
  ```
- 删除之前通过文件或名称创建的资源：

  ```sh
  kubectl delete -f my-app-deployment.yaml
  # 或者通过名称删除特定的 pod
  kubectl delete pod my-app-pod-12345
  ```

`kubectl` 是您控制集群的入口。随着学习的深入，您将从简单的检查命令转向管理复杂的应用程序定义。掌握此工具是实现高效 Kubernetes 管理的第一步。直接使用命令创建资源与在文件中定义资源之间的区别，将带我们进入一个主题：声明式管理模型。

## 参考资料

- [kubectl Overview](https://kubernetes.io/docs/reference/kubectl/overview/) — Kubernetes Authors (2024)
  官方文档，介绍 `kubectl` 的作用、与 API 服务器的通信方式以及基本用法。
- [Organize Cluster Access Using kubeconfig Files](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) — Kubernetes Authors (2024)
  官方文档，详细说明 `kubeconfig` 文件的结构及其管理，用于集群访问和身份验证。
- [kubectl Command Reference](https://kubernetes.io/docs/reference/kubectl/) — Kubernetes Authors (2024)
  《kubectl》命令的综合指南，包括其语法、参数和实际示例。

---

[上一节](03-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E5%B7%A5%E4%BD%9C%E8%8A%82%E7%82%B9.md) · [下一节](05-%E5%A3%B0%E6%98%8E%E5%BC%8F%E4%B8%8E%E6%8C%87%E4%BB%A4%E5%BC%8F%E7%AE%A1%E7%90%86.md)
