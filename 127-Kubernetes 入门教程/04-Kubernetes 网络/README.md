# 第 4 章：Kubernetes 网络

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes)

[返回课程目录](../README.md)

在前面的章节中，你学会了使用 Pod 运行应用程序，并通过 Deployment 管理其生命周期。然而，Pod 是暂时的，其 IP 地址并不固定，这给通信带来了挑战。应用程序的各个组件如何可靠地相互发现并进行对话？你又该如何让集群外部的用户访问你的程序？本章主要讲解旨在解决这些问题的 Kubernetes 网络模型。

你将先学习集群内部通信的原理。接着，我们会介绍 `Service` 对象，这是一种为一组 Pod 提供持久网络端点的抽象方式。你将学会区分并使用各种 Service 类型，包括 `ClusterIP`、`NodePort` 和 `LoadBalancer`，从而控制程序的公开方式。最后，我们会讲解 `Ingress`，这是一种管理集群内服务外部 HTTP 和 HTTPS 访问的资源。

## 小节

- 1. [Kubernetes 网络模型](01-Kubernetes%20%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B.md)
- 2. [Service 简介](02-Service%20%E7%AE%80%E4%BB%8B.md)
- 3. [Service 类型：ClusterIP、NodePort 与 LoadBalancer](03-Service%20%E7%B1%BB%E5%9E%8B%EF%BC%9AClusterIP%E3%80%81NodePort%20%E4%B8%8E%20LoadBalancer.md)
- 4. [定义 Service 清单](04-%E5%AE%9A%E4%B9%89%20Service%20%E6%B8%85%E5%8D%95.md)
- 5. [集群内的服务发现](05-%E9%9B%86%E7%BE%A4%E5%86%85%E7%9A%84%E6%9C%8D%E5%8A%A1%E5%8F%91%E7%8E%B0.md)
- 6. [通过 Ingress 暴露服务](06-%E9%80%9A%E8%BF%87%20Ingress%20%E6%9A%B4%E9%9C%B2%E6%9C%8D%E5%8A%A1.md)
- 7. [动手实践：暴露应用程序](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9A%B4%E9%9C%B2%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md)
