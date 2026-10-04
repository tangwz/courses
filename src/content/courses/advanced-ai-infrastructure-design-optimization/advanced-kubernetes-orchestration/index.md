---
course: "advanced-ai-infrastructure-design-optimization"
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration"
sourceId: 1224
chapter: "advanced-kubernetes-orchestration"
title: "第三章：使用Kubernetes进行高级资源编排"
order: 3
description: "掌握用于管理机器学习工作负载的Kubernetes高级用法，包括GPU共享、竞价实例处理和集群自动扩缩。"
hasQuiz: false
---

尽管上一章侧重于单个分布式训练任务的运行方式，但生产系统必须并发高效地管理许多此类任务。直接在云虚拟机上操作，在调度、故障容错和资源利用方面存在难题。容器编排提供了必要的抽象层来解决这些问题。

本章将详细说明如何使用Kubernetes编排大规模机器学习工作负载。您将配置生产级功能，以在共享计算集群上管理模型的整个生命周期。我们首先使用 KubeFlow 定义并自动化机器学习流水线。接着，您将学习通过高级GPU调度来管理专用硬件，这包括时间切片和多实例GPU（MIG）配置。随后，我们将通过设置集群自动扩缩来使计算资源供应与工作负载需求匹配，并通过制定使用低成本竞价实例的策略来提升运行效率。本章最后介绍实现多租户的方法，使多个团队能够安全地共享基础设施并拥有明确的资源边界。
