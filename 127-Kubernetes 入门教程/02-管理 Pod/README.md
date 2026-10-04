# 第 2 章：管理 Pod

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods)

[返回课程目录](../README.md)

Pod 是 Kubernetes 中调度的原子单位。它代表集群中运行的进程，作为一个或多个容器及其共享存储和网络资源的封装。由于所有容器化应用都在 Pod 中运行，学会如何创建和管理它们是使用 Kubernetes 的必备技能。

本章提供了管理 Pod 的技术指南。我们将从 Pod 抽象本身以及如何使用 YAML 清单定义其规范开始。你将学会创建单容器和多容器 Pod，了解 Pod 经历的生命周期阶段，并使用 `kubectl` 进行查看和调试等常用操作。最后，我们将配置存活（liveness）和就绪（readiness）探针，帮助 Kubernetes 自动管理应用程序的健康状况。

## 小节

- 1. [Pod 抽象层](01-Pod%20%E6%8A%BD%E8%B1%A1%E5%B1%82.md)
- 2. [在 YAML 中编写 Pod 清单](02-%E5%9C%A8%20YAML%20%E4%B8%AD%E7%BC%96%E5%86%99%20Pod%20%E6%B8%85%E5%8D%95.md)
- 3. [单容器与多容器 Pod](03-%E5%8D%95%E5%AE%B9%E5%99%A8%E4%B8%8E%E5%A4%9A%E5%AE%B9%E5%99%A8%20Pod.md)
- 4. [Pod 的生命周期](04-Pod%20%E7%9A%84%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F.md)
- 5. [使用 kubectl 进行 Pod 操作](05-%E4%BD%BF%E7%94%A8%20kubectl%20%E8%BF%9B%E8%A1%8C%20Pod%20%E6%93%8D%E4%BD%9C.md)
- 6. [健康检查：存活探针与就绪探针](06-%E5%81%A5%E5%BA%B7%E6%A3%80%E6%9F%A5%EF%BC%9A%E5%AD%98%E6%B4%BB%E6%8E%A2%E9%92%88%E4%B8%8E%E5%B0%B1%E7%BB%AA%E6%8E%A2%E9%92%88.md)
- 7. [练习：部署与检查 Pod](07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E9%83%A8%E7%BD%B2%E4%B8%8E%E6%A3%80%E6%9F%A5%20Pod.md)
