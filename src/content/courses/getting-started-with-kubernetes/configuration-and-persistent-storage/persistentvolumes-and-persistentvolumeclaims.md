---
course: "getting-started-with-kubernetes"
chapter: "configuration-and-persistent-storage"
lesson: "persistentvolumes-and-persistentvolumeclaims"
sourceId: 7326
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/persistentvolumes-and-persistentvolumeclaims"
title: "PersistentVolumes 和 PersistentVolumeClaims"
description: "了解管理员如何配置存储 (PV) 以及用户如何请求存储 (PVC)，从而实现独立于 Pod 的存储生命周期管理。"
order: 4
plots: []
sourceHash: "8675fa119ff74a4af6591c6c71aefce592889ad015ab054232f3b27b3e1ea888"
sourceCorrections: []
---

为了解决容器存储生命周期短的问题，Kubernetes 提供了一种抽象机制，将存储基础设施的管理与应用程序的使用解耦。这是通过两个既有区别又相互关联的 API 对象实现的：`PersistentVolumes`（持久卷）和 `PersistentVolumeClaims`（持久卷声明）。这种责任划分允许集群管理员管理存储资源，而开发者则可以请求这些资源，无需了解底层的实现细节。

### PersistentVolumes：存储的供应

`PersistentVolume` (PV) 是集群中由管理员预先配置的一块存储。它是集群中的资源，就像节点（Node）是集群资源一样。PV 是类似于 `Volumes` 的卷插件，但其生命周期独立于任何使用它的 Pod。此 API 对象捕获了存储实现的详细信息，无论它是 NFS 共享、iSCSI 卷，还是特定云服务商提供的存储系统（如 AWS EBS 或 GCE Persistent Disk）。

管理员创建一个 PV 来代表实际的后端存储。在定义 `PersistentVolume` 时，需要指定其特性，例如容量、访问模式和回收策略。

- **`capacity`**：该卷包含的存储总量，例如 `10Gi`。
- **`accessModes`**：定义卷的挂载方式。这是一个必须由底层存储提供商支持的属性。
  - `ReadWriteOnce` (RWO)：卷可以被单个节点以读写模式挂载。
  - `ReadOnlyMany` (ROX)：卷可以被多个节点以只读模式挂载。
  - `ReadWriteMany` (RWX)：卷可以被多个节点以读写模式挂载。
- **`persistentVolumeReclaimPolicy`**：当与其绑定的声明被删除后，PV 会发生什么。
  - `Retain`（保留）：卷不会被自动清理。数据保留，卷必须由管理员手动回收。这是最安全且通常是默认的策略。
  - `Delete`（删除）：卷以及外部基础设施中关联的存储资产将被删除。
  - `Recycle`（回收）：（已废弃）对卷执行基本的擦除操作，使其可再次用于新的声明。

下面是一个 `PersistentVolume` 清单示例，它使用工作节点上的本地目录配置了一个 5Gi 的卷。这种 `hostPath` 类型适用于 Minikube 等单节点开发集群。

```yaml
# pv-definition.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: task-pv-volume
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/mnt/data"
```

创建后，PV 存在于集群中并处于 `Available`（可用）状态，等待被 `PersistentVolumeClaim` 申领。

### PersistentVolumeClaims：存储的请求

`PersistentVolumeClaim` (PVC) 是开发者或用户发出的存储请求。如果说 PV 是存储的供应，那么 PVC 就是需求。它类似于 Pod。Pod 消耗节点资源（如 CPU 和内存）；而 PVC 消耗 PV 资源。

开发者通过创建 PVC 来指定应用程序所需的存储量和访问模式，而无需了解底层存储基础设施。

下面是一个 `PersistentVolumeClaim` 的清单，它请求的存储与我们上面定义的 PV 兼容。

```yaml
# pvc-definition.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: task-pvc-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
```

创建此 PVC 时，Kubernetes 控制平面会寻找满足声明要求的可用 PV。在本例中，我们的 5Gi PV 满足了 3Gi 的请求，并且匹配 `ReadWriteOnce` 访问模式。如果找到匹配项，控制平面就会将 PV 与 PVC 绑定。PV 的状态随后从 `Available` 变为 `Bound`（已绑定）。

> 此图说明了 Kubernetes 存储对象之间的关系。Pod 引用 `PersistentVolumeClaim` 来请求存储。PVC 绑定到合适的 `PersistentVolume`，而 PV 反过来代表由基础设施管理的物理存储资源。

一旦声明与卷绑定，它就是独占绑定的。这种绑定是 `PersistentVolume` 和 `PersistentVolumeClaim` 之间的一一映射。这种机制确保了没有其他 PVC 可以申领同一个 PV。

### 使用 StorageClasses 实现自动配置

手动创建 `PersistentVolumes` 适用于小规模或特殊设置，但难以扩展。在大多数生产环境（尤其是云端）中，存储是动态配置的。这在 Kubernetes 中是通过名为 `StorageClass` 的对象来处理的。

`StorageClass` 为管理员提供了一种描述其提供的存储“类”的方法。不同的类可以映射到服务质量级别、备份策略或由集群管理员确定的任意策略。

创建 PVC 时，可以指定 `storageClassName`。如果指定了，`StorageClass` 配置的制备器（provisioner）将自动创建一个匹配的 PV 并将其绑定到该 PVC。

例如，云提供商可能会提供一个名为 `gp2` 的 `StorageClass` 用于通用 SSD 存储，以及另一个名为 `st1` 的用于吞吐量 (throughput)优化的 HDD 存储。开发者随后只需请求所需的存储类型：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-database-pvc
spec:
  storageClassName: gp2 # 请求特定类型的存储
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

创建此 PVC 后，云提供商的存储制备器会自动创建一个 100Gi 的 `gp2` 卷，在 Kubernetes 中为其创建相应的 PV 对象，并将其绑定到 `my-database-pvc` 声明。

PV 和 PVC 机制提供了一种管理存储资源的有效方式。管理员可以管理底层存储资源池，而开发者可以通过简单的请求来使用存储。了解了如何申领持久化存储后，下一步就是学习如何将这些声明挂载到 Pod 中，以便应用程序能够使用这些存储。

## 参考资料

- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) — Kubernetes Authors (2024)
  Kubernetes 官方文档，解释持久卷和持久卷声明、其生命周期及核心属性。
- [Kubernetes Up and Running: Dive into the Future of Infrastructure](https://play.google.com/store/books/details/Kubernetes_Up_and_Running_Dive_into_the_Future_of?id=v-7GEAAAQBAJ) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media; Pages: 326
  一本全面书籍，涵盖 Kubernetes 基础知识，包括持久化存储和有状态应用程序管理的详细章节。
