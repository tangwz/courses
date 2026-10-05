# 使用 ConfigMap 管理配置

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/managing-configuration-with-configmaps)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接将配置设置存储在容器镜像中会导致系统僵化且耦合度过高。每当需要更改数据库端点、调整功能开关或修改外部 API URL 时，都必须重新构建、推送并重新部署整个镜像。这个过程不仅缓慢、容易出错，而且违反了配置与代码分离的原则。Kubernetes 提供了一个专门的对象来解决这个问题：ConfigMap。

ConfigMap 是一种 Kubernetes API 对象，用于以键值对的形式存储非机密配置数据。可以将其看作是集群内部的一个字典或哈希表，供应用程序访问。通过将配置从容器镜像中解耦，你可以管理不同环境（开发、测试、生产）的设置，而无需更改应用程序的核心产物。

### 创建 ConfigMap

主要有两种创建 ConfigMap 的方式：使用 `kubectl` 的命令式创建，或使用 YAML 清单文件的声明式创建。

#### 使用 kubectl 进行命令式创建

`kubectl create configmap` 命令适用于快速、即时的创建。你可以直接从字面量或文件中提供数据。

要从字面量键值对创建 ConfigMap，请使用 `--from-literal` 标志：

```bash
kubectl create configmap app-settings \
  --from-literal=api.service.url=http://users-api:8080 \
  --from-literal=ui.theme=dark
```

此命令创建了一个名为 `app-settings` 的 ConfigMap，其中包含两个数据项。你可以通过 `kubectl describe configmap app-settings` 查看其详细信息。

要从整个文件创建 ConfigMap，可以使用 `--from-file` 标志。这对于现有的配置文件（如 `nginx.conf` 或 `application.properties`）特别有用。

假设你有一个名为 `app.properties` 的文件：

```properties
# 应用程序配置
feature.new-login.enabled=true
cache.size.mb=256
```

你可以创建一个 ConfigMap，将此文件的内容存储在与文件名对应的键下：

```bash
kubectl create configmap app-file-config --from-file=app.properties
```

#### 使用 YAML 清单进行声明式创建

为了实现可重现且受版本控制的基础设施，使用 YAML 清单的声明式方法是标准做法。ConfigMap 清单定义了期望状态，你可以将其应用到集群中。

以下是 `my-configmap.yaml` 中定义的 ConfigMap 示例：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-settings-v2
  namespace: default
data:
  # 以属性形式定义的键值对
  database.host: "mysql.prod.svc.cluster.local"
  database.name: "products"
  
  # 具有多行内容作为单个值的键
  ui.properties: |
    color.primary=#4263eb
    font.size=14px
```

通过应用此清单来创建对象：

```bash
kubectl apply -f my-configmap.yaml
```

`data` 字段包含键值对。键必须是有效的 DNS 子域名，值是任意的 UTF-8 字符串。你可以使用管道符 `|` 来表示多行字符串值，这有助于将整个配置文件直接嵌入 (embedding)到清单中。

### 将 ConfigMap 数据注入 Pod

创建 ConfigMap 后，你需要一种方式让应用程序 Pod 使用其数据。Kubernetes 提供了三种主要方法：作为环境变量注入、作为卷中的文件挂载，或作为命令行参数 (parameter)使用。

#### 将 ConfigMap 作为环境变量使用

将配置数据作为环境变量注入是一种常见模式，因为许多应用程序都设计为从环境中读取设置。

你可以使用 `envFrom` 将 ConfigMap 中的所有键值对注入 Pod：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  containers:
  - name: my-app-container
    image: nginx:latest
    envFrom:
    - configMapRef:
        name: app-settings
```

在此示例中，`app-settings` ConfigMap 中的每个键都会变成 `my-app-container` 内部的一个环境变量。例如，键 `api.service.url` 将变成名为 `API_SERVICE_URL` 的环境变量。请注意，Kubernetes 会按照标准环境变量命名约定，将键转换为大写名称，并将点和横杠替换为下划线。

或者，你可以使用 `valueFrom` 注入 ConfigMap 中的特定键：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod-selective
spec:
  containers:
  - name: my-app-container
    image: busybox
    command: [ "/bin/sh", "-c", "echo Theme is $APP_THEME" ]
    env:
    - name: APP_THEME
      valueFrom:
        configMapKeyRef:
          name: app-settings # ConfigMap 的名称
          key: ui.theme      # 要注入的键
```

这种方法提供了更精细的控制，允许你选择要公开哪些键以及如何命名生成的环境变量。

#### 将 ConfigMap 作为卷使用

对于期望从文件中读取配置的应用程序，将 ConfigMap 挂载为卷是理想的方案。当你挂载 ConfigMap 时，`data` 字段中的每个键都会变成挂载目录中的一个文件，文件的内容就是该键的值。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod-volume
spec:
  containers:
  - name: my-app-container
    image: nginx
    ports:
    - containerPort: 80
    volumeMounts:
    - name: config-volume   # 引用下方定义的卷
      mountPath: /etc/config # 容器内的目录
  volumes:
  - name: config-volume     # 定义卷
    configMap:
      name: app-settings-v2 # 要挂载的 ConfigMap
```

在此清单中：

1. Pod 级别的 `volumes` 部分定义了一个名为 `config-volume` 的卷，并指定其来源为 `app-settings-v2` ConfigMap。
2. 容器规范中的 `volumeMounts` 部分将该卷挂载到容器文件系统的 `/etc/config` 路径下。

Pod 启动后，你可以在容器内执行 shell 并发现以下文件结构：

- `/etc/config/database.host`（包含字符串 `mysql.prod.svc.cluster.local`）
- `/etc/config/database.name`（包含字符串 `products`）
- `/etc/config/ui.properties`（包含 ConfigMap 中定义的多行字符串）

此方法非常有效，因为它允许你在不更改代码的情况下为应用程序提供完整的配置文件。

> 图示显示了一个 ConfigMap 被两个不同的 Pod 使用。Pod A 以环境变量的形式注入数据，而 Pod B 将相同的数据挂载为卷中的文件。

### 更新如何传播

理解对 ConfigMap 的更改如何影响正在运行的 Pod 是一个重点。传播方式取决于 ConfigMap 的使用方式。

- **环境变量：** 如果 Pod 将 ConfigMap 作为环境变量使用，这些值是在 Pod 启动时注入的。对 ConfigMap 的更新**不会**反映在正在运行的容器中。环境变量在容器的生命周期内是静态的。要应用新配置，必须重新启动 Pod。
- **挂载卷：** 如果 Pod 将 ConfigMap 作为挂载卷使用，当 ConfigMap 更改时，挂载的文件会自动更新。这种同步由 Kubelet 管理，可能会有短暂延迟。然而，容器内运行的应用程序必须能够检测到文件更改并重新加载其配置。许多应用程序不会自动执行此操作，因此可能仍需要重新启动才能获取新设置。

由于这种特性，推出配置更改最可靠的方法是触发管理 Pod 的 Deployment 进行滚动更新。这可以确保以受控方式创建带有更新配置的新 Pod，从而防止不一致性。

通过使用 ConfigMap，你可以在 Kubernetes 上构建更灵活且易于管理的应用程序。该对象在应用逻辑与其运行时配置之间提供了清晰的分离，这是创建可移植和可扩展系统的基本实践。

## 参考资料

- [Configure a Pod to Use a ConfigMap](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/) — Kubernetes Authors (2024)
  Journal: Kubernetes.io
  Kubernetes官方文档，详细介绍了如何创建、使用和管理ConfigMap，以便将非敏感配置数据注入到Pod中。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure, 3rd Edition](https://www.oreilly.com/library/view/kubernetes-up-and/9781098110154/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本介绍Kubernetes核心概念的书籍，涵盖了使用ConfigMap进行配置管理以及云原生应用部署的最佳实践。

---

[上一节](../04-Kubernetes%20%E7%BD%91%E7%BB%9C/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9A%B4%E9%9C%B2%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20Secret%20%E5%A4%84%E7%90%86%E5%87%AD%E6%8D%AE.md)
