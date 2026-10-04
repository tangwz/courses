# 定义 Deployment 清单

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/defining-deployment-manifest)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Deployment 提供了一种在 Kubernetes 中管理应用程序期望状态的方法。Deployment 清单是一个描述这种期望状态的 YAML 文件。你不需要分步骤告诉 Kubernetes “要做什么”，而是告诉它你“想要什么”结果。随后，Deployment 控制器会执行相关工作，使集群的当前状态与你声明的状态保持一致。

这种方式更加可靠且可预测。清单作为应用程序部署的版本控制文档，定义了从容器镜像到运行副本数量以及更新策略的所有内容。

### Deployment 清单的结构

Deployment 清单在 Pod 清单结构的基础上，增加了用于管理副本和更新的字段。让我们来看一个典型的 Nginx Web 服务器清单。

```yaml
# nginx-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
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

该清单包含四个必需的顶级字段：`apiVersion`、`kind`、`metadata` 和 `spec`。

- **`apiVersion: apps/v1`**：这指定了用于创建此对象的 Kubernetes API 版本。Deployment 属于 `apps` API 组，`v1` 是其稳定版本。
- **`kind: Deployment`**：这确定了要创建的对象类型。
- **`metadata`**：这包含有助于唯一标识对象的数据，例如它的 `name`（名称）和描述性的 `labels`（标签）。虽然 Deployment 对象本身的标签用于组织管理，但 `spec` 内部的标签则具有实际的功能作用。
- **`spec`**：你可以在此处定义 Deployment 的期望状态。这是最核心的部分，包含了应用程序运行方式的逻辑。

### Deployment 规范 (`spec`)

`spec` 字段是 Deployment 的重心，详细说明了控制器的指令。它由三个主要字段组成：`replicas`、`selector` 和 `template`。

#### `replicas`

这个字段很好理解。它指定了你希望运行的 Pod 期望数量。在我们的示例中，`replicas: 3` 告诉 Kubernetes 确保始终有三个符合此应用程序模板的 Pod 在运行。如果某个 Pod 发生故障，控制器会自动创建一个新 Pod 以维持三个副本的数量。

#### `selector`

`selector`（选择器）字段告诉 Deployment 控制器要管理哪些 Pod。它通过匹配 `matchLabels` 中定义的标签来实现这一点。在我们的示例中，选择器是 `{ "matchLabels": { "app": "nginx" } }`。这意味着 Deployment 将查找并管理任何带有 `app: nginx` 标签的 Pod。

控制器选择器与 Pod 标签之间的这种关联非常基础。选择器必须与 `template` 部分中分配给 Pod 的标签相匹配。如果它们不匹配，Deployment 会创建 Pod 但无法找到它们，从而导致不断创建 Pod 的死循环。

#### `template`

`template` 字段 (`spec.template`) 是 Deployment 将要创建的 Pod 的蓝图。其内容与独立的 Pod 清单几乎完全相同，包含自己的 `metadata` 和 `spec`。

- **`template.metadata.labels`**：这些是应用于由 Deployment 创建的每个 Pod 的标签。这些标签必须与 Deployment 的 `spec.selector.matchLabels` 匹配。在我们的示例中，模板中的 `app: nginx` 与选择器匹配，从而将 Pod 正确关联到 Deployment。
- **`template.spec`**：这定义了将在每个 Pod 内部运行的容器规范，就像标准的 Pod 清单一样。你可以指定容器的 `name`（名称）、要使用的容器 `image`（镜像）以及容器将暴露的任何 `ports`（端口）。

下图说明了这些组件是如何连接的。Deployment 并不直接管理 Pod，而是管理一个 ReplicaSet，由 ReplicaSet 进而确保正确数量的 Pod 正在运行。选择器是连接这三个对象的机制。

> Deployment 创建并管理一个 ReplicaSet。Deployment 规范中定义的选择器被 ReplicaSet 用于识别和管理 Pod，而这些 Pod 是根据 Pod 模板创建的。

当你使用 `kubectl apply -f nginx-deployment.yaml` 之类的命令应用此清单时，Kubernetes 控制平面会读取它并创建一个 Deployment 对象。随后，Deployment 控制器创建一个相应的 ReplicaSet，而 ReplicaSet 控制器则创建那三个 Pod。从那时起，控制器会持续运行，以确保集群状态符合此定义。

## 参考资料

- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — Kubernetes Authors (N/A)
  Kubernetes Deployments的官方文档，解释其规范以及如何管理期望的应用状态，包括与ReplicaSet和Pod的交互。
- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications
  一本综合性书籍，提供Kubernetes组件的实践见解和详细解释，包括Deployment及其相关控制器的结构和功能。
- [Kubernetes Controllers](https://kubernetes.io/docs/concepts/architecture/controller/) — Kubernetes Authors (N/A)
  Publisher: Kubernetes Project
  Kubernetes控制器架构作用的官方文档，控制器为Deployment等对象实现声明式期望状态管理。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20Deployment%20%E8%BF%9B%E8%A1%8C%E5%BA%94%E7%94%A8%E5%8F%91%E5%B8%83.md) · [下一节](05-%E6%89%A7%E8%A1%8C%20Deployment%20%E6%9B%B4%E6%96%B0%E4%B8%8E%E5%9B%9E%E6%BB%9A.md)
