---
course: "getting-started-with-kubernetes"
chapter: "configuration-and-persistent-storage"
lesson: "handling-credentials-with-secrets"
sourceId: 7324
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/handling-credentials-with-secrets"
title: "使用 Secret 处理凭据"
description: "了解如何使用 Secret 存储和管理敏感信息，例如密码、OAuth 令牌和 SSH 密钥。"
order: 2
plots: []
sourceHash: "2eee1089aa72e7a90002673b74dda1dccf1f2e171840d96358738da48088a868"
sourceCorrections: []
---

在 Kubernetes 中运行的应用程序通常需要访问敏感信息，例如 API 密钥、数据库凭据或 TLS 证书。虽然管理非敏感配置数据有专门的工具，但将敏感数据直接放在 Pod 清单或容器镜像中会带来很大的安全风险。为了解决这个问题，Kubernetes 提供了一个专门的对象：`Secret`。

`Secret` 在结构上与 `ConfigMap` 类似。它是一个以键值对形式存储少量敏感数据的对象。主要的区别在于用途以及 Kubernetes 处理数据的方式。Secret 专为机密信息设计，集群可以为其提供额外的保护，例如避免将其写入日志，并支持在 etcd 中启用静态加密。

### Secret 的构成

Secret 将敏感数据与 Pod 定义解耦。这让你可以根据安全最佳实践，独立管理凭据和应用程序代码。默认情况下，Secret 中的数据是经过 base64 编码的。需要明确的是，**base64 是一种编码格式，而不是加密机制**。任何能够访问编码字符串的人都可以轻松地对其进行解码。Secret 真正的安全优势来自于它与 Kubernetes 基于角色的访问控制（RBAC）的整合，这限制了哪些用户或服务账户可以读取 Secret 对象本身。

Kubernetes 支持几种类型的 Secret，通过清单中的 `type` 字段指定。默认且最常用的类型是 `Opaque`，用于存储任意的用户自定义数据。其他类型适用于特定的场景，例如用于存储私有容器镜像仓库凭据的 `kubernetes.io/dockerconfigjson`。对于大多数应用程序凭据，你会使用 `Opaque` 类型。

### 创建 Secret

你可以使用命令式的 `kubectl` 命令或声明式的 YAML 清单来创建 Secret。

#### 使用 kubectl 进行命令式创建

对于快速任务或简单的凭据，`kubectl create secret` 非常高效。最常用的变体是 `generic`，它会创建一个 `Opaque` 类型的 Secret。

要创建一个包含用户名和密码的名为 `db-credentials` 的 Secret，可以使用 `--from-literal` 参数 (parameter)。

```shell
kubectl create secret generic db-credentials \
  --from-literal=username=webapp_user \
  --from-literal=password='P@s5w0rd!_123'
```

此命令会在当前命名空间中创建一个新的 Secret。如果你运行 `kubectl get secret db-credentials -o yaml`，你会看到 Kubernetes 已经自动对你提供的字面值进行了 base64 编码。

#### 使用清单进行声明式创建

对于生产环境和版本控制的基础设施（GitOps），在清单中声明式地定义 Secret 是标准做法。

在 YAML 中定义 Secret 时，`data` 字段中的值**必须经过 base64 编码**。你可以从命令行生成编码字符串。

例如，对密码进行编码：

```shell
# -n 参数很重要，可以防止 echo 添加换行符
echo -n 'P@s5w0rd!_123' | base64
```

输出将是 `UEBzNXcwcmQhXzEyMw==`。然后你就可以在清单中使用这个编码后的值。

下面是同一个 `db-credentials` Secret 的清单：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  username: d2ViYXBwX3VzZXI= # "webapp_user" 编码后的值
  password: UEBzNXcwcmQhXzEyMw== # "P@s5w0rd!_123" 编码后的值
```

手动编码数值可能比较麻烦。Kubernetes 提供了 `stringData` 字段作为替代方案。使用 `stringData` 时，你可以提供原始的、未编码的字符串，Kubernetes 会在创建 Secret 时为你处理编码。`stringData` 字段是只写的；之后查看 Secret 时它是不可见的。

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-key-secret
type: Opaque
stringData:
  # Kubernetes 会自动编码这个值
  apiKey: "abC123XyZ-Your-Super-Secret-Api-Key"
```

### 在 Pod 中使用 Secret

创建 Secret 后，Pod 可以通过两种主要方式使用其数据：作为环境变量，或者作为挂载到卷中的文件。

> 图示说明了 Pod 使用 Secret 数据的两种主要方式。

#### 将 Secret 注入为环境变量

将 Secret 注入为环境变量是设计为从环境读取配置的应用程序常用的模式。你可以在容器定义的 `env` 或 `envFrom` 字段中使用它。

要从 Secret 中注入特定的键，请使用 `valueFrom`：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  containers:
  - name: my-app
    image: nginx
    env:
      - name: DATABASE_USER
        valueFrom:
          secretKeyRef:
            name: db-credentials  # Secret 的名称
            key: username        # Secret 中的键
      - name: DATABASE_PASSWORD
        valueFrom:
          secretKeyRef:
            name: db-credentials
            key: password
```

在这个例子中，容器将拥有两个环境变量 `DATABASE_USER` 和 `DATABASE_PASSWORD`，它们的值由 `db-credentials` Secret 中解码后的内容填充。

#### 将 Secret 挂载为卷

对于包含多个值或代表整个配置文件（如 TLS 私钥或 JSON 服务账户密钥）的 Secret，将其挂载为卷更为合适。当 Secret 挂载为卷时，Secret 中的每个键值对都会成为挂载目录中的一个文件。文件名是键，文件内容是解码后的值。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod-volume
spec:
  containers:
  - name: my-app
    image: nginx
    volumeMounts:
    - name: secret-volume
      mountPath: "/etc/app-secrets"
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: db-credentials
```

在该 Pod 启动后，容器内会有一个 `/etc/app-secrets` 目录。该目录下将有两个文件：`username` 和 `password`，包含相应的凭据值。使用 `readOnly: true` 标志是一个好的习惯，可以防止应用程序意外修改 Secret 文件。

### 安全措施

虽然 Secret 是 Kubernetes 中处理凭据的指定方式，但遵循安全最佳实践仍然非常重要：

1. **使用 RBAC 限制访问：** Secret 的主要防御手段是强大的 RBAC 策略。仅向绝对需要 Secret 的用户、组或服务账户授予对 Secret 的 `get`、`list` 和 `watch` 权限。
2. **启用静态加密：** 默认情况下，etcd 中的 Secret 数据是不加密的。在生产集群中，你应该配置静态加密，以防攻击者在获得 etcd 备份的读取权限时保护你的 Secret 数据。
3. **使用外部 Secret 存储：** 对于更高的安全要求，可以考虑集成外部 Secret 管理器，如 HashiCorp Vault 或 AWS Secrets Manager。这些系统提供集中管理、审计和动态 Secret 生成。可以使用 Kubernetes Operator 将这些外部存储中的 Secret 同步到集群中。
4. **避免将清单提交到 Git：** 不要将包含 base64 编码数据的原始 Secret 清单提交到版本控制系统中。在提交之前，使用 Bitnami Sealed Secrets 或 SOPS (Secrets OPerationS) 等工具对 Secret 清单进行加密。

## 参考资料

- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) — Kubernetes Authors (2024)
  解释Kubernetes Secrets的用途、结构、创建和使用方式的官方文档。
- [Protecting Secret Data](https://kubernetes.io/docs/concepts/configuration/secret/#protecting-secret-data) — Kubernetes Authors (2024)
  Publisher: Kubernetes
  详细说明保护Secret数据的安全实践，包括静态加密和RBAC配置。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running/9781491935668/) — Kelsey Hightower, Brendan Burns, Joe Beda (2019)
  Publisher: O'Reilly Media
  一本理解Kubernetes架构和操作模式的基础书籍，内容涵盖配置管理和安全。
- [Bitnami Sealed Secrets](https://github.com/bitnami/sealed-secrets) — Bitnami (2024)
  一个Kubernetes控制器和工具，用于加密Secret，使其可以安全地存储在Git中。
