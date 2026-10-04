---
course: "deploying-diffusion-models-scale"
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment"
sourceId: 996
chapter: "infrastructure-scalable-deployment"
title: "可扩展部署架构"
order: 3
description: "构建可扩展的部署架构，使用容器、Kubernetes 和云端 GPU/无服务器资源。"
hasQuiz: false
---

在对扩散模型进行推理优化之后，下一步是构建其能够高效进行大规模运行的运行环境。本章将专注于构建必要的支撑体系。我们将介绍如何使用 Docker 等容器打包模型及其依赖项、使用 Kubernetes 等编排工具管理部署和扩展，以及配置云资源，包括 GPU 等专用硬件和无服务器计算选项。

主要议题包括管理容器内的 GPU 资源、根据推理需求实现自动扩展，以及处理大型模型和生成数据时的存储考量。您将学习设计并实现能够处理可变负载同时管理计算资源的系统，专门处理 GPU 密集型扩散模型的需求。实践练习将指导您在 Kubernetes 集群上部署容器化模型。
