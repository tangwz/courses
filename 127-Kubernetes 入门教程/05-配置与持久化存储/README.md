# 第 5 章：配置与持久化存储

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage)

[返回课程目录](../README.md)

应用程序通常需要外部配置，以及一个能够存放持久化数据的地方，以确保数据在容器实例销毁后依然存在。尽管 Deployment 和 Pod 能有效管理应用逻辑，但它们本身并不能解决配置文件、凭证或持久化数据的管理问题。

本章介绍为这些需求设计的 Kubernetes 对象。我们首先来看应用配置，学习如何将其与容器镜像分离。你将学会使用 `ConfigMaps` 处理非敏感数据，并使用 `Secrets` 管理数据库密码或 `$API_KEY` 等敏感信息。

接下来，我们将讨论数据持久化。容器的文件系统是临时的，一旦容器停止，其中的内容就会丢失。为了管理有状态应用，需要一种将数据存储在可靠位置的方法。我们将了解 Kubernetes 的存储架构，特别是 `PersistentVolumes` (PV) 和 `PersistentVolumeClaims` (PVC)，它们允许应用程序申请并连接到稳定的存储资源。

完成本章学习后，你将能够为应用注入配置并提供持久化存储，从而掌握在 Kubernetes 集群中管理有状态服务所需的各项技能。

## 小节

- 1. [使用 ConfigMap 管理配置](01-%E4%BD%BF%E7%94%A8%20ConfigMap%20%E7%AE%A1%E7%90%86%E9%85%8D%E7%BD%AE.md)
- 2. [使用 Secret 处理凭据](02-%E4%BD%BF%E7%94%A8%20Secret%20%E5%A4%84%E7%90%86%E5%87%AD%E6%8D%AE.md)
- 3. [Kubernetes 存储概念](03-Kubernetes%20%E5%AD%98%E5%82%A8%E6%A6%82%E5%BF%B5.md)
- 4. [PersistentVolumes 和 PersistentVolumeClaims](04-PersistentVolumes%20%E5%92%8C%20PersistentVolumeClaims.md)
- 5. [将存储卷挂载到 Pod](05-%E5%B0%86%E5%AD%98%E5%82%A8%E5%8D%B7%E6%8C%82%E8%BD%BD%E5%88%B0%20Pod.md)
- 6. [实践：注入配置并挂载卷](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B3%A8%E5%85%A5%E9%85%8D%E7%BD%AE%E5%B9%B6%E6%8C%82%E8%BD%BD%E5%8D%B7.md)
