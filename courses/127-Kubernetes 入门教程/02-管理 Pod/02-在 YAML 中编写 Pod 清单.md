# 在 YAML 中编写 Pod 清单

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods/writing-pod-manifest-yaml)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要在 Kubernetes 中创建 Pod，你需要在名为“清单”（manifest）的 YAML 文件中定义其期望状态。该文件充当蓝图，告诉 Kubernetes 控制平面你具体想要运行什么。虽然你可以使用命令行参数 (parameter)以命令式方式创建 Pod，但在生产环境中，使用清单的声明式方法是标准做法，因为它支持版本控制、可重复性，并且更易于管理配置。

Pod 清单由几个必需的顶级字段组成。让我们看看定义任何 Pod 时都会用到的核心组件。

### Pod 清单的结构

每个 Kubernetes 对象清单（包括 Pod）都包含四个主要字段：`apiVersion`、`kind`、`metadata` 和 `spec`。

> Kubernetes Pod 清单的层级结构，显示了定义容器所需的字段和嵌套配置。

1. **apiVersion**: 此字段指定你用于创建此对象的 Kubernetes API 版本。对于 Pod、Service 和 ConfigMap 等核心对象，版本为 `v1`。对于 Deployment 或 ReplicaSet 等其他对象，你将使用不同的 API 版本，例如 `apps/v1`。API 版本决定了该对象合法的结构和字段。
2. **kind**: 此字段指定你要创建的 Kubernetes 对象类型。要创建 Pod，请将此值设置为 `Pod`。这会告诉 Kubernetes 该清单描述的是哪种资源。
3. **metadata**: 这是一个包含有助于唯一标识资源的字段的对象。`metadata` 中最常用的两个字段是：

   - `name`: Pod 在其命名空间内的唯一名称。例如 `webapp-pod-xyz`。
   - `labels`: 附加到对象的键值对映射。标签用于组织管理，并允许其他资源（如 Service）选择并操作一组 Pod。例如，你可以为所有 Web 服务器 Pod 打上 `app: frontend` 标签。
4. **spec** (规格): 你在此字段中定义对象的*期望状态*。不同类型的 Kubernetes 对象，`spec` 的内容差异很大。对于 Pod，`spec` 主要描述应在其中运行的容器。`spec` 中最核心的字段是 `containers`。

### 定义容器

`spec.containers` 字段是一个列表，这意味着一个 Pod 可以包含一个或多个容器。列表中的每一项都是一个具有自己属性集的容器对象。你至少必须提供：

- **name**: Pod 内容器的名称。这对于查看日志和识别容器非常有用，尤其是在多容器 Pod 中。
- **image**: 要运行的容器镜像，指定格式与 Docker 相同，例如 `nginx:1.21.6` 或 `your-repo/your-app:v2`。Kubernetes 将从容器镜像仓库中拉取此镜像。
- **ports**: 要从容器暴露的网络端口列表。`containerPort` 指定容器内应用程序正在监听的端口号。请注意，这不会将端口暴露到 Pod 外部；它仅供其他 Kubernetes 组件参考。将应用程序暴露给网络需要使用 Service，我们将在后续章节中介绍。

### 示例：单容器 Pod 清单

让我们将这些部分组合成一个完整的清单文件。本示例定义了一个名为 `nginx-pod` 的简单 Pod，它运行一个使用官方 Nginx 镜像的容器。

创建一个名为 `nginx-pod.yaml` 的文件并添加以下内容：

```yaml
# nginx-pod.yaml

# 指定 API 版本。'v1' 是核心 API 组。
apiVersion: v1

# 指定对象类型。在本例中是 Pod。
kind: Pod

# 包含 Pod 的元数据，如名称和标签。
metadata:
  # 此 Pod 在命名空间内的唯一名称。
  name: nginx-pod
  # 标签是用于组织和选择的键值对。
  labels:
    app: webserver

# 定义 Pod 的期望状态。
spec:
  # Pod 内部运行的容器列表。
  containers:
    # 列表中的每一项都是一个容器定义。
  - name: nginx-container
    # 用于此容器的 Docker 镜像。
    image: nginx:1.21.6
    # 从容器暴露的端口列表。
    ports:
    - containerPort: 80
```

此清单是一条声明式指令。它并不说明*如何*创建 Pod，而是说明 Pod 的最终状态应该是*什么*。当你将此清单应用于集群时，Kubernetes 控制平面会读取它，并努力使集群的实际状态与此期望状态相匹配。如果名为 `nginx-pod` 的 Pod 不存在，它将被创建。

要应用此清单，你可以使用 `kubectl` 命令行工具，我们将在本章稍后部分详细介绍：

```bash
kubectl apply -f nginx-pod.yaml
```

此命令将清单发送到 Kubernetes API 服务器，然后由其调度 Pod 在工作节点上运行。有了对清单结构的这些基础了解，你现在已经准备好定义简单或复杂的 Pod 配置了。

## 参考资料

- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) — Kubernetes Authors (2024)
  提供关于Kubernetes Pod的官方和最新信息，包括其生命周期、结构以及通过manifest进行声明式定义。
- [Kubernetes API Concepts](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) — Kubernetes Authors (2024)
  解释所有Kubernetes API对象的基本结构，详细阐述apiVersion、kind、metadata和spec字段的用途。
- [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action) — Marko Lukša (2017)
  Publisher: Manning Publications
  全面介绍了Kubernetes，其中包含专门章节解释Pod、YAML manifest以及资源管理的声明式方法。

---

[上一节](01-Pod%20%E6%8A%BD%E8%B1%A1%E5%B1%82.md) · [下一节](03-%E5%8D%95%E5%AE%B9%E5%99%A8%E4%B8%8E%E5%A4%9A%E5%AE%B9%E5%99%A8%20Pod.md)
