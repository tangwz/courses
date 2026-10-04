---
course: "getting-started-with-kubernetes"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals"
sourceId: 1268
chapter: "kubernetes-fundamentals"
title: "Kubernetes 基础知识"
order: 1
description: "学习 Kubernetes 架构、控制平面与节点组件、kubectl，以及如何使用 Minikube 或 Kind 搭建本地 Kubernetes 集群。"
hasQuiz: false
---

运行单个容器已不再是难题，但管理由多个容器组成的分布式应用会带来不小的运维难度。像 Kubernetes 这样的容器编排器正是为了解决这些难题而设计的。本章介绍使用 Kubernetes 的架构与操作基础，帮助你建立关于系统构造及交互方式的功能模型。

你将从了解 Kubernetes 的核心架构组件开始。内容包括：

*   **控制平面：** 负责集群全局决策的组件，例如 API Server、调度器和 etcd。
*   **工作节点：** 运行在每个节点上以维护 Pod 运行的组件，包括 Kubelet 和容器运行时。
*   **`kubectl`：** 用于与 Kubernetes API 通信的命令行工具。
*   **声明式管理：** 与运行命令式指令相比，这种做法通过 YAML 文件定义期望状态。

为了将理论付诸实践，本章将指导你搭建本地 Kubernetes 集群。最后通过动手练习，你将使用 `kubectl` 检查集群组件，在开始部署应用前巩固所学内容。
