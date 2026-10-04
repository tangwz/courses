---
course: "getting-started-with-kubernetes"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes"
sourceId: 1271
chapter: "networking-in-kubernetes"
title: "Kubernetes 网络"
order: 4
description: "学习 Kubernetes 的网络工作原理。内容包括网络模型、服务发现，以及如何通过 Service 和 Ingress 对外发布应用程序。"
hasQuiz: false
---

在前面的章节中，你学会了使用 Pod 运行应用程序，并通过 Deployment 管理其生命周期。然而，Pod 是暂时的，其 IP 地址并不固定，这给通信带来了挑战。应用程序的各个组件如何可靠地相互发现并进行对话？你又该如何让集群外部的用户访问你的程序？本章主要讲解旨在解决这些问题的 Kubernetes 网络模型。

你将先学习集群内部通信的原理。接着，我们会介绍 `Service` 对象，这是一种为一组 Pod 提供持久网络端点的抽象方式。你将学会区分并使用各种 Service 类型，包括 `ClusterIP`、`NodePort` 和 `LoadBalancer`，从而控制程序的公开方式。最后，我们会讲解 `Ingress`，这是一种管理集群内服务外部 HTTP 和 HTTPS 访问的资源。
