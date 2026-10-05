# 第 3 章：使用控制器管理工作负载

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers)

[返回课程目录](../README.md)

尽管 Pod 是 Kubernetes 中的基本执行单元，但逐个管理它们并不是运行应用程序的高效方案。如果节点出现故障，独立的 Pod 不会被重新调度，且扩缩容需要你手动逐个创建或删除 Pod。这种手动操作方式效率较低，且无法提供现代容器编排器所应具备的抗风险能力。

为了实现工作负载管理的自动化，Kubernetes 使用了 **控制器**。控制器是一种主动调节机制，负责跟踪至少一种 Kubernetes 资源类型。这些对象包含一个表示期望状态的 spec 字段。控制器的任务是使当前状态与期望状态保持一致。

本章介绍管理无状态应用工作负载的控制器。我们将学习如何：

*   使用 **ReplicaSet** 确保指定数量的 Pod 副本始终运行。
*   使用 **Deployment** 以声明式方式管理应用的发布与更新。
*   执行滚动更新，在不停机的情况下部署应用新版本，并在需要时回滚到之前的版本。

学完本章后，你将不再局限于管理单个 Pod，而是学会使用更高层级的抽象来管理整个应用生命周期，从而实现扩缩容、自愈和自动更新。

## 小节

- 1. [Kubernetes 控制器简介](01-Kubernetes%20%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%80%E4%BB%8B.md)
- 2. [使用 ReplicaSet 确保 Pod 可用性](02-%E4%BD%BF%E7%94%A8%20ReplicaSet%20%E7%A1%AE%E4%BF%9D%20Pod%20%E5%8F%AF%E7%94%A8%E6%80%A7.md)
- 3. [使用 Deployment 进行应用发布](03-%E4%BD%BF%E7%94%A8%20Deployment%20%E8%BF%9B%E8%A1%8C%E5%BA%94%E7%94%A8%E5%8F%91%E5%B8%83.md)
- 4. [定义 Deployment 清单](04-%E5%AE%9A%E4%B9%89%20Deployment%20%E6%B8%85%E5%8D%95.md)
- 5. [执行 Deployment 更新与回滚](05-%E6%89%A7%E8%A1%8C%20Deployment%20%E6%9B%B4%E6%96%B0%E4%B8%8E%E5%9B%9E%E6%BB%9A.md)
- 6. [检查 Deployment 状态](06-%E6%A3%80%E6%9F%A5%20Deployment%20%E7%8A%B6%E6%80%81.md)
- 7. [动手实践：创建与更新 Deployment](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E4%B8%8E%E6%9B%B4%E6%96%B0%20Deployment.md)
