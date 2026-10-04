---
course: "planning-optimizing-ai-infrastructure"
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml"
sourceId: 1227
chapter: "containerization-orchestration-for-ml"
title: "机器学习的容器化与编排"
order: 4
description: "使用Docker创建可重现的机器学习环境，并使用Kubernetes通过Kubeflow编排可扩展的训练和部署管道。"
hasQuiz: false
---

当您从本地开发转向生产系统时，确保您的机器学习代码在不同环境中一致运行成为一个重要难题。操作系统、库版本或硬件驱动程序的差异可能导致难以调试的错误。本章通过介绍用于创建可移植且可扩展的机器学习工作流的工具来解决此问题。

您将从容器化行业标准Docker开始。我们将介绍如何将应用程序及其依赖项和配置打包成一个称为容器的独立单元。您将学习如何专门为机器学习应用程序编写 `Dockerfile`，包括必要的CUDA和Python库。接下来，我们将转向使用Kubernetes进行编排。您将了解Kubernetes如何自动化容器化应用程序的部署、扩展和管理，使其成为处理复杂机器学习工作负载的有效平台。各部分将详细介绍如何在Kubernetes集群中管理GPU资源，并引入Kubeflow以构建结构化的机器学习管道。本章最后将通过一个实践练习，您将在其中容器化并部署一个模型服务应用程序。
