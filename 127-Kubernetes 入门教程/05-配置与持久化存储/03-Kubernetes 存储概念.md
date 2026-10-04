# Kubernetes 存储概念

来源：[原文](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage/kubernetes-storage-concepts)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 Pod 为容器提供了运行环境，但其内部文件系统是临时的。当 Pod 停止运行时，其中的数据也会随之消失。对于任何需要保持状态的应用（从简单的数据库到机器学习 (machine learning)模型检查点），这种临时性都构成了巨大的挑战。Kubernetes 通过一套分层的存储抽象来解决这个问题，这些抽象可以独立于 Pod 生命周期来管理数据持久化。

### Volume（卷）抽象

在最基础的层面上，Kubernetes 使用 `Volume` 来管理存储。Volume 是一个目录，其中可能包含一些数据，Pod 中的容器可以访问该目录。Volume 的定义特征是其生命周期与所属的 Pod 直接挂钩。它在 Pod 被调度到节点上时创建，并只要该 Pod 存在就一直存在。

这意味着如果 Pod 中的容器重启，Volume 的内容会被保留。这对于在多容器 Pod 之间共享数据，或者在容器崩溃后保留数据非常有用。例如，简单的 `emptyDir` 类型的 Volume 提供了一个临时的临时空间，它在 Pod 分配到节点时初始化，在 Pod 被移除时删除。

但是，如果 Pod 本身被删除并替换，`emptyDir` 也会被销毁。虽然这种模式很有用，但它并没有解决长期数据持久化的问题。为此，我们需要一种将存储生命周期与 Pod 生命周期解耦的机制。

### 使用 PersistentVolume 和 Claim 实现存储解耦

为了有效地支持有状态应用，开发人员不应该需要了解底层存储基础设施的细节。无论存储是由 Google Persistent Disk 等云服务商提供的，还是像 NFS 这样的网络连接系统，亦或是本地存储阵列，都应该是实现层面的细节。

Kubernetes 通过两个主要的 API 对象实现了这种关注点分离：

- **PersistentVolume (PV)**：集群中的一块存储，由管理员预先配置，或使用 `StorageClass` 动态配置。PV 是集群中的一种资源，就像 CPU 或内存资源一样。它的生命周期独立于任何使用它的单个 Pod。
- **PersistentVolumeClaim (PVC)**：用户或开发人员对存储的请求。PVC 是开发人员消耗存储资源的方式。它指定了所需的容量大小和访问模式（例如，是一次只能由一个节点挂载，还是可以由多个节点以只读方式挂载）。

这种交互建立了一个清晰的职责划分。集群管理员通过创建 `PersistentVolume` 对象来配置和管理可用存储池。开发人员在无需了解底层存储技术的情况下，创建 `PersistentVolumeClaim` 为其应用请求存储。随后，Kubernetes 控制平面会寻找满足该 Claim 要求的合适 PV，并将两者绑定在一起。

一旦 PVC 绑定到了 PV，该 PV 就会为该 Claim 保留，不能再被其他 Claim 绑定。开发人员可以定义一个引用该 PVC 的 Pod，Kubernetes 会将底层存储挂载到 Pod 的容器中。

> Pod、PersistentVolumeClaim (PVC) 和 PersistentVolume (PV) 之间的关系。开发人员的 Pod 通过 PVC 请求存储，Kubernetes 将其绑定到管理员配置的可用 PV 上。

### 使用 StorageClass 进行动态配置

为每个存储请求手动创建 `PersistentVolume` 对象可能效率低下，特别是在大型多租户集群中。为了自动完成这个过程，Kubernetes 引入了另一个对象：`StorageClass`。

`StorageClass` 为管理员提供了一种定义不同存储“类型”的方法。每种类型可能对应不同的服务质量、备份策略或底层存储系统。例如，管理员可以定义 `fast-ssd`（由高性能 SSD 支持）和 `standard-hdd`（由机械硬盘支持）等类别。

当开发人员创建 `PersistentVolumeClaim` 时，可以指定 `StorageClass` 名称。集群中该 `StorageClass` 的置备程序（provisioner）会自动为该请求创建一个新的 PV，而不是将其绑定到现有的、手动创建的 PV 上。这个过程称为动态配置，它消除了管理员预先配置存储的需求，实现了存储的按需创建。

通过 Volume、PersistentVolume/Claim 和 StorageClass 这三种抽象，Kubernetes 提供了一个灵活且强大的存储管理框架。接下来的章节将演示如何定义这些对象并将持久化存储附加到应用中。

## 参考资料

- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) — Kubernetes Authors (2024)
  Kubernetes 持久化存储概念的基本指南，涵盖 PV、PVC 和 StorageClass。
- [Kubernetes: Up and Running: Dive into the Future of Infrastructure, Fourth Edition](https://www.vitalsource.com/products/kubernetes-up-and-running-brendan-burns-v9781098110154) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media; Pages: 320
  一本全面介绍 Kubernetes 架构和操作方面的书籍，包含专门的存储管理章节。
- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications
  提供了 Kubernetes 核心概念的实用、深入解释，包含关于持久化存储和卷管理的详细章节。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20Secret%20%E5%A4%84%E7%90%86%E5%87%AD%E6%8D%AE.md) · [下一节](04-PersistentVolumes%20%E5%92%8C%20PersistentVolumeClaims.md)
