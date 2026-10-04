---
course: "getting-started-with-kubernetes"
chapter: "configuration-and-persistent-storage"
lesson: "practice-injecting-config-mounting-volume"
sourceId: 7328
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/practice-injecting-config-mounting-volume"
title: "实践：注入配置并挂载卷"
description: "通过动手实践课程创建 ConfigMap 和 Secret 并将其注入 Pod，同时为有状态数据附加持久卷。"
order: 6
plots: []
sourceHash: "c773863bc9e2c98c48efc90fdb28dbaf0b81b00880499b06ac57dc540614da86"
sourceCorrections: []
---

本动手练习将引导你为应用程序设置创建 `ConfigMap`，为凭据创建 `Secret`，并为持久存储创建 `PersistentVolumeClaim`。接着，你将部署一个同时使用这三者的 Pod，以此演示如何管理一个完整的、有状态的应用程序组件。

进行此实践需要一个运行中的 Kubernetes 集群（例如通过 Minikube 或 Kind 运行的集群），并配置好 `kubectl` 以与其交互。

### 第 1 步：创建 ConfigMap 和 Secret

首先，我们将创建要注入到应用程序 Pod 中的配置对象。我们将从用于非敏感数据的 `ConfigMap` 开始，然后是用于模拟 API 密钥的 `Secret`。

1. **创建 ConfigMap：**
   使用 `kubectl create configmap` 命令，从直接量生成一个名为 `app-config` 的 `ConfigMap`。该 `ConfigMap` 将保存一个简单的属性文件格式。

   ```sh
   kubectl create configmap app-config \
     --from-literal=app.properties='
   greeting=Hello
   log.level=INFO
   '
   ```
2. **创建 Secret：**
   接下来，创建一个名为 `api-credentials` 的 `Secret`。我们将再次使用 `--from-literal` 标志来提供一个模拟的 API 密钥。请记住，虽然 Kubernetes 将 Secret 存储为 base64 编码的字符串，但它们在 etcd 中默认并未加密，应被视为敏感信息。

   ```sh
   kubectl create secret generic api-credentials \
     --from-literal=API_KEY='abc-123-def-456'
   ```
3. **验证创建情况：**
   你可以使用 `kubectl get` 和 `kubectl describe` 查看刚刚创建的对象。例如，要查看 `ConfigMap`：

   ```sh
   kubectl get configmap app-config -o yaml
   ```

   你将看到提供的数据存储在 `data` 字段下。

### 第 2 步：申请持久存储

配置就绪后，下一步是申请存储。我们通过创建 `PersistentVolumeClaim` (PVC) 来实现这一点。在大多数本地开发集群和云服务商中，都会配置一个默认的 `StorageClass`，用于动态地分配满足申请要求的 `PersistentVolume` (PV)。

1. **定义 PersistentVolumeClaim：**
   创建一个名为 `pvc.yaml` 的文件，内容如下。此清单申请一个 1Gi 的小型卷，访问模式为 `ReadWriteOnce`，这意味着它可以被单个节点以读写方式挂载。

   ```yaml
   # pvc.yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: my-data-claim
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 1Gi
   ```
2. **应用并验证 PVC：**
   将清单应用到你的集群。

   ```sh
   kubectl apply -f pvc.yaml
   ```

   现在，检查 PVC 的状态。它应该很快从 `Pending` 转换为 `Bound`，这表明 `PersistentVolume` 已成功分配并与其绑定。

   ```sh
   kubectl get pvc my-data-claim
   ```

   你应该会看到类似这样的输出：

   ```
   NAME            STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
   my-data-claim   Bound    pvc-a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6   1Gi        RWO            standard       15s
   ```

### 第 3 步：部署挂载了卷和配置的 Pod

现在我们将所有内容整合在一起。下图显示了我们要创建的 Pod 与其依赖的配置和存储资源之间的关系。

> `stateful-app` Pod 将 `Secret` 用作环境变量，将 `ConfigMap` 挂载为文件，并挂载 `PersistentVolumeClaim` 以提供持久的文件系统。

1. **定义 Pod：**
   创建一个名为 `stateful-pod.yaml` 的文件。此清单定义了一个 Pod，它：

   - 将 `api-credentials` Secret 中的所有键值对作为环境变量注入。
   - 定义了两个卷：一个用于 `ConfigMap`，另一个用于 `PersistentVolumeClaim`。
   - 将 `ConfigMap` 卷挂载在 `/etc/config`，使 `app.properties` 键可以作为文件使用。
   - 将 PVC 卷挂载在 `/data` 以进行持久存储。

   ```yaml
   # stateful-pod.yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: stateful-app
   spec:
     containers:
     - name: app-container
       image: alpine
       # 保持容器运行
       command: ["/bin/sh", "-c", "sleep 3600"]
       envFrom:
       - secretRef:
           name: api-credentials
       volumeMounts:
       - name: config-volume
         mountPath: /etc/config
       - name: data-volume
         mountPath: /data
     volumes:
     - name: config-volume
       configMap:
         name: app-config
     - name: data-volume
       persistentVolumeClaim:
         claimName: my-data-claim
   ```

   我们使用简单的 `alpine` 镜像和 `sleep` 命令来保持容器运行，以便我们可以检查它。
2. **部署 Pod：**
   将 Pod 清单应用到你的集群。

   ```sh
   kubectl apply -f stateful-pod.yaml
   ```

### 第 4 步：验证配置和数据持久性

让我们确认配置和存储是否已正确附加到运行中的 Pod。我们将使用 `kubectl exec` 在容器内运行命令。

1. **验证来自 Secret 的环境变量：**
   在 Pod 内部执行命令以打印其环境变量，并过滤我们的 API 密钥。

   ```sh
   kubectl exec stateful-app -- env | grep API_KEY
   ```

   输出应为 `API_KEY=abc-123-def-456`，确认 Secret 已成功注入。
2. **验证来自 ConfigMap 的挂载文件：**
   检查从 `ConfigMap` 挂载的文件内容。

   ```sh
   kubectl exec stateful-app -- cat /etc/config/app.properties
   ```

   这将显示我们之前定义的内容：`greeting=Hello` 和 `log.level=INFO`。
3. **测试数据持久性：**
   现在，向挂载在 `/data` 的持久存储卷写入一个文件。

   ```sh
   kubectl exec stateful-app -- sh -c "echo '有状态数据在重启后依然存在' > /data/message.txt"
   ```

   为了证明数据是持久的，删除该 Pod 并重新创建它。`PersistentVolumeClaim` 和底层的 `PersistentVolume` 不会受到影响。

   ```sh
   kubectl delete pod stateful-app
   kubectl apply -f stateful-pod.yaml
   ```

   等待新的 Pod 进入 `Running` 状态。你可以使用 `kubectl get pod stateful-app` 检查其状态。一旦它运行起来，尝试读取你之前创建的文件。

   ```sh
   kubectl exec stateful-app -- cat /data/message.txt
   ```

   该命令将输出 `有状态数据在重启后依然存在`。这证实了即使 Pod 被完全销毁并替换，写入卷的数据仍然存在。

### 清理

为了保持集群整洁，请删除本次实践中创建的资源。

```sh
kubectl delete pod stateful-app
kubectl delete pvc my-data-claim
kubectl delete secret api-credentials
kubectl delete configmap app-config
```

你现在已经成功管理了应用程序的配置和有状态数据，这是在 Kubernetes 上运行生产级负载的两项基本技能。

## 参考资料

- [Configure a Pod to Use a ConfigMap](https://kubernetes.io/docs/tasks/configure-pod-container/configure-a-pod-configmap/) — Kubernetes Authors (2025)
  Publisher: The Linux Foundation
  关于如何创建和使用 ConfigMap 将配置数据注入 Pod 的官方指南。
- [Configure a Pod to Use a Secret](https://kubernetes.io/docs/concepts/configuration/secret/#using-secrets-as-environment-variables) — Kubernetes Authors (2024)
  关于如何创建和使用 Secret 来处理敏感数据（包括作为环境变量注入）的官方指南。
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) — Kubernetes Authors (2024)
  关于持久卷和持久卷声明的官方概念及实践指南，对有状态应用至关重要。
