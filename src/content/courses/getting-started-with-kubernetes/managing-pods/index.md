---
course: "getting-started-with-kubernetes"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-2-managing-pods"
sourceId: 1269
chapter: "managing-pods"
title: "管理 Pod"
order: 2
description: "学习如何定义、创建和管理 Kubernetes Pod，这是 Kubernetes 中最小的可部署单元。内容涉及 YAML 清单、生命周期和健康探针。"
hasQuiz: false
---

Pod 是 Kubernetes 中调度的原子单位。它代表集群中运行的进程，作为一个或多个容器及其共享存储和网络资源的封装。由于所有容器化应用都在 Pod 中运行，学会如何创建和管理它们是使用 Kubernetes 的必备技能。

本章提供了管理 Pod 的技术指南。我们将从 Pod 抽象本身以及如何使用 YAML 清单定义其规范开始。你将学会创建单容器和多容器 Pod，了解 Pod 经历的生命周期阶段，并使用 `kubectl` 进行查看和调试等常用操作。最后，我们将配置存活（liveness）和就绪（readiness）探针，帮助 Kubernetes 自动管理应用程序的健康状况。
