---
course: "getting-started-with-kubernetes"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers"
sourceId: 1270
chapter: "workload-management-controllers"
title: "使用控制器管理工作负载"
order: 3
description: "使用 Kubernetes 控制器管理应用的伸缩性与弹性。了解用于发布的 Deployment 以及保障可用性的 ReplicaSet。"
hasQuiz: false
---

尽管 Pod 是 Kubernetes 中的基本执行单元，但逐个管理它们并不是运行应用程序的高效方案。如果节点出现故障，独立的 Pod 不会被重新调度，且扩缩容需要你手动逐个创建或删除 Pod。这种手动操作方式效率较低，且无法提供现代容器编排器所应具备的抗风险能力。

为了实现工作负载管理的自动化，Kubernetes 使用了 **控制器**。控制器是一种主动调节机制，负责跟踪至少一种 Kubernetes 资源类型。这些对象包含一个表示期望状态的 spec 字段。控制器的任务是使当前状态与期望状态保持一致。

本章介绍管理无状态应用工作负载的控制器。我们将学习如何：

*   使用 **ReplicaSet** 确保指定数量的 Pod 副本始终运行。
*   使用 **Deployment** 以声明式方式管理应用的发布与更新。
*   执行滚动更新，在不停机的情况下部署应用新版本，并在需要时回滚到之前的版本。

学完本章后，你将不再局限于管理单个 Pod，而是学会使用更高层级的抽象来管理整个应用生命周期，从而实现扩缩容、自愈和自动更新。
