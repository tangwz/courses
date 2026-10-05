# 第 1 章：Kubernetes 基础知识

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals)

[返回课程目录](../README.md)

运行单个容器已不再是难题，但管理由多个容器组成的分布式应用会带来不小的运维难度。像 Kubernetes 这样的容器编排器正是为了解决这些难题而设计的。本章介绍使用 Kubernetes 的架构与操作基础，帮助你建立关于系统构造及交互方式的功能模型。

你将从了解 Kubernetes 的核心架构组件开始。内容包括：

*   **控制平面：** 负责集群全局决策的组件，例如 API Server、调度器和 etcd。
*   **工作节点：** 运行在每个节点上以维护 Pod 运行的组件，包括 Kubelet 和容器运行时。
*   **`kubectl`：** 用于与 Kubernetes API 通信的命令行工具。
*   **声明式管理：** 与运行命令式指令相比，这种做法通过 YAML 文件定义期望状态。

为了将理论付诸实践，本章将指导你搭建本地 Kubernetes 集群。最后通过动手练习，你将使用 `kubectl` 检查集群组件，在开始部署应用前巩固所学内容。

## 小节

- 1. [什么是容器编排](01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%AE%B9%E5%99%A8%E7%BC%96%E6%8E%92.md)
- 2. [Kubernetes 架构：控制平面](02-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E6%8E%A7%E5%88%B6%E5%B9%B3%E9%9D%A2.md)
- 3. [Kubernetes 架构：工作节点](03-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E5%B7%A5%E4%BD%9C%E8%8A%82%E7%82%B9.md)
- 4. [kubectl 的作用](04-kubectl%20%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 5. [声明式与指令式管理](05-%E5%A3%B0%E6%98%8E%E5%BC%8F%E4%B8%8E%E6%8C%87%E4%BB%A4%E5%BC%8F%E7%AE%A1%E7%90%86.md)
- 6. [搭建本地 Kubernetes 集群](06-%E6%90%AD%E5%BB%BA%E6%9C%AC%E5%9C%B0%20Kubernetes%20%E9%9B%86%E7%BE%A4.md)
- 7. [动手实践：集群检查](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%9B%86%E7%BE%A4%E6%A3%80%E6%9F%A5.md)
