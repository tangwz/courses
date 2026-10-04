---
course: "getting-started-with-kubernetes"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-5-configuration-and-persistent-storage"
sourceId: 1272
chapter: "configuration-and-persistent-storage"
title: "配置与持久化存储"
order: 5
description: "学习使用 ConfigMaps 和 Secrets 管理应用配置，并通过 Volume 和 PVC 为有状态应用提供持久化存储。"
hasQuiz: false
---

应用程序通常需要外部配置，以及一个能够存放持久化数据的地方，以确保数据在容器实例销毁后依然存在。尽管 Deployment 和 Pod 能有效管理应用逻辑，但它们本身并不能解决配置文件、凭证或持久化数据的管理问题。

本章介绍为这些需求设计的 Kubernetes 对象。我们首先来看应用配置，学习如何将其与容器镜像分离。你将学会使用 `ConfigMaps` 处理非敏感数据，并使用 `Secrets` 管理数据库密码或 `$API_KEY` 等敏感信息。

接下来，我们将讨论数据持久化。容器的文件系统是临时的，一旦容器停止，其中的内容就会丢失。为了管理有状态应用，需要一种将数据存储在可靠位置的方法。我们将了解 Kubernetes 的存储架构，特别是 `PersistentVolumes` (PV) 和 `PersistentVolumeClaims` (PVC)，它们允许应用程序申请并连接到稳定的存储资源。

完成本章学习后，你将能够为应用注入配置并提供持久化存储，从而掌握在 Kubernetes 集群中管理有状态服务所需的各项技能。
