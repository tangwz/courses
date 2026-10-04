# 将存储卷挂载到 Pod

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/attaching-volumes-to-pods)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Kubernetes 资源（如 `ConfigMap`、`Secret` 或 `PersistentVolumeClaim` 对象）可以在集群中创建并存在。但是，要将它们与应用程序关联，必须将其作为存储卷（Volume）挂载到 Pod。这种挂载过程需要在 Pod 清单中进行两部分声明：首先在 Pod 级别定义存储卷，然后将其挂载到特定容器内的目标位置。

这种机制在不同类型的存储卷中是一致的，为将外部数据和配置连接到应用程序负载提供了标准方法。为 Pod 定义的存储卷可以由其所有容器共享，也可以单独挂载到每个容器中。

### 两步走流程：`volumes` 和 `volumeMounts`

将存储卷连接到 Pod 内部的容器始终涉及清单中的两个不同规范：

1. **`spec.volumes`**：位于 Pod 规范级别的数组。你可以在此处声明 Pod 可用的存储卷列表。此列表中的每个条目在 Pod 内部都有一个唯一的 `name`（名称），并指定其来源，例如 `ConfigMap`、`Secret` 或 `PersistentVolumeClaim`。可以将其理解为创建一个指向外部资源的命名指针。
2. **`spec.containers.volumeMounts`**：位于容器规范内部的数组。此列表告诉容器要挂载 Pod `volumes` 列表中的哪些存储卷，以及将其放置在其自身文件系统的什么位置。它使用 `volumes` 列表中的 `name` 来识别来源，并为目标位置指定 `mountPath`（挂载路径）。

`name` 字段起到了桥梁作用，将 Pod 级别的存储卷定义与容器级别的挂载点连接起来。

> `spec.volumes` 数组中的 `name` 字段被容器 `volumeMounts` 数组中的 `name` 字段引用，从而建立连接。

让我们看看如何针对已讨论的每种资源类型应用此模式。

### 挂载 ConfigMap

`ConfigMaps` 常用于向应用程序提供配置文件。通过将 `ConfigMap` 挂载为存储卷，`ConfigMap` 中的每个数据键（key）都会成为指定挂载目录下的一个文件，键名作为文件名，键值作为文件内容。

假设你有以下用于配置 Web 服务器的 `ConfigMap`：

```yaml
# my-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # 'settings.json' 将成为文件名
  settings.json: |
    {
      "theme": "dark",
      "logLevel": "info"
    }
  # 'user.properties' 也会成为一个文件
  user.properties: |
    username=default_user
    allow_guests=false
```

要将此 `ConfigMap` 挂载到 Pod 中，你可以编写如下清单：

```yaml
# my-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: configmap-pod
spec:
  containers:
  - name: my-app
    image: busybox
    command: ["sleep", "3600"]
    # 第 2 步：将存储卷挂载到容器的文件系统中
    volumeMounts:
    - name: config-volume # 此名称必须与下方的 volume 名称匹配
      mountPath: /etc/config # 文件将出现的目录

  # 第 1 步：在 Pod 级别定义存储卷
  volumes:
  - name: config-volume # 此存储卷在 Pod 内的唯一名称
    configMap:
      # 指定要使用的 ConfigMap 名称
      name: app-config
```

应用此清单后，你可以检查 Pod 的文件系统：

```shell
# 进入正在运行的 Pod
kubectl exec -it configmap-pod -- /bin/sh

# 列出挂载路径的内容
/ # ls /etc/config
settings.json
user.properties

# 查看其中一个文件的内容
/ # cat /etc/config/settings.json
{
  "theme": "dark",
  "logLevel": "info"
}
```

`settings.json` 和 `user.properties` 文件现在出现在容器内的 `/etc/config` 目录中。这种方法成功实现了配置与应用程序容器镜像的分离。

### 挂载 Secret

挂载 `Secret` 的步骤与挂载 `ConfigMap` 几乎完全相同。这种一致性使得管理敏感数据和非敏感数据变得简单。主要的区别在于将存储卷来源指定为 `secret` 而不是 `configMap`。

假设你有一个包含凭据的 `Secret`：

```yaml
# my-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData: # 为了可读性使用 stringData
  username: "admin"
  password: "SuperSecretPassword123"
```

你可以按如下方式将此 `Secret` 挂载到 Pod：

```yaml
# secret-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secret-pod
spec:
  containers:
  - name: my-app
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true # 将 Secret 挂载为只读是良好的实践

  volumes:
  - name: secret-volume
    secret:
      # 指定要使用的 Secret 名称
      secretName: db-credentials
```

当此 Pod 运行时，`Secret` 中的键（`username` 和 `password`）将在 `/etc/secrets` 目录中作为文件提供。这些文件的内容将是 `Secret` 中解码后的值。

```shell
# 进入正在运行的 Pod
kubectl exec -it secret-pod -- /bin/sh

# 列出 Secret 挂载目录中的文件
/ # ls /etc/secrets
password
username

# 显示密码
/ # cat /etc/secrets/password
SuperSecretPassword123
```

### 挂载 PersistentVolumeClaim

对于需要持久存储的有状态应用程序，你需要挂载 `PersistentVolumeClaim` (PVC)。这使你的 Pod 能够访问独立于 Pod 生命周期存在的持久磁盘。Pod 清单引用的是 PVC，而不是底层的 `PersistentVolume` (PV)，从而保持了用户存储请求与管理员配置存储之间的抽象关系。

假设已经创建了一个名为 `my-data-pvc` 的 `PersistentVolumeClaim` 且处于 `Bound` 状态，你可以像这样将其挂载到 Pod：

```yaml
# stateful-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: stateful-app
spec:
  containers:
  - name: web-server
    image: nginx
    ports:
    - containerPort: 80
    volumeMounts:
    - name: data-storage # 引用下方定义的存储卷
      mountPath: /usr/share/nginx/html # 网页内容的挂载路径

  volumes:
  - name: data-storage # 存储卷名称
    persistentVolumeClaim:
      # 引用 PVC 的名称
      claimName: my-data-pvc
```

在此示例中，由 `my-data-pvc` 管理的存储被挂载到容器的 `/usr/share/nginx/html` 路径下。Nginx 服务器写入该目录的任何文件都将存储在持久卷上。如果 Pod 被删除并重新创建，新 Pod 可以挂载相同的 PVC 并重新获得对之前写入数据的访问权限。这是在 Kubernetes 上运行数据库、内容管理系统和其他有状态负载的基础。

> Pod 的 `volumes` 部分引用不同的集群资源，而每个容器中的 `volumeMounts` 部分则将它们映射到特定的路径。

通过掌握这种定义存储卷和创建存储卷挂载的模式，你就能管理广泛的应用程序需求，从具有外部配置的无状态服务，到依赖持久存储的复杂有状态应用程序。

## 参考资料

- [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/) — Kubernetes Authors (2024)
  介绍Kubernetes卷的核心概念，包括ConfigMap、Secret和PersistentVolumeClaim卷类型以及如何使用spec.volumes和volumeMounts。
- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/) — Kubernetes Authors (2024)
  详细说明如何使用ConfigMap来外部化配置数据，包括将其作为卷挂载到Pod中。
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) — Kubernetes Authors (2024)
  提供使用Secret管理敏感数据的信息，包括将其作为卷挂载的最佳实践。
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) — Kubernetes Authors (2024)
  描述Kubernetes中持久存储的工作原理，重点介绍PersistentVolumes和PersistentVolumeClaims及其在Pod中的应用。

---

[上一节](04-PersistentVolumes%20%E5%92%8C%20PersistentVolumeClaims.md) · [下一节](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B3%A8%E5%85%A5%E9%85%8D%E7%BD%AE%E5%B9%B6%E6%8C%82%E8%BD%BD%E5%8D%B7.md)
