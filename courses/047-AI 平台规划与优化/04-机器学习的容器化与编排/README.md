# 第 4 章：机器学习的容器化与编排

来源：[原章节](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml)

[返回课程目录](../README.md)

当您从本地开发转向生产系统时，确保您的机器学习代码在不同环境中一致运行成为一个重要难题。操作系统、库版本或硬件驱动程序的差异可能导致难以调试的错误。本章通过介绍用于创建可移植且可扩展的机器学习工作流的工具来解决此问题。

您将从容器化行业标准Docker开始。我们将介绍如何将应用程序及其依赖项和配置打包成一个称为容器的独立单元。您将学习如何专门为机器学习应用程序编写 `Dockerfile`，包括必要的CUDA和Python库。接下来，我们将转向使用Kubernetes进行编排。您将了解Kubernetes如何自动化容器化应用程序的部署、扩展和管理，使其成为处理复杂机器学习工作负载的有效平台。各部分将详细介绍如何在Kubernetes集群中管理GPU资源，并引入Kubeflow以构建结构化的机器学习管道。本章最后将通过一个实践练习，您将在其中容器化并部署一个模型服务应用程序。

## 小节

- 1. [Docker 在可复现环境中的使用介绍](01-Docker%20%E5%9C%A8%E5%8F%AF%E5%A4%8D%E7%8E%B0%E7%8E%AF%E5%A2%83%E4%B8%AD%E7%9A%84%E4%BD%BF%E7%94%A8%E4%BB%8B%E7%BB%8D.md)
- 2. [构建包含机器学习库的Docker镜像](02-%E6%9E%84%E5%BB%BA%E5%8C%85%E5%90%AB%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BA%93%E7%9A%84Docker%E9%95%9C%E5%83%8F.md)
- 3. [Kubernetes 管理机器学习工作负载简介](03-Kubernetes%20%E7%AE%A1%E7%90%86%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E7%AE%80%E4%BB%8B.md)
- 4. [Kubernetes 组件：Pod、Service、Deployment](04-Kubernetes%20%E7%BB%84%E4%BB%B6%EF%BC%9APod%E3%80%81Service%E3%80%81Deployment.md)
- 5. [在Kubernetes集群中管理GPU资源](05-%E5%9C%A8Kubernetes%E9%9B%86%E7%BE%A4%E4%B8%AD%E7%AE%A1%E7%90%86GPU%E8%B5%84%E6%BA%90.md)
- 6. [使用 Kubeflow 构建机器学习管道](06-%E4%BD%BF%E7%94%A8%20Kubeflow%20%E6%9E%84%E5%BB%BA%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%A1%E9%81%93.md)
- 7. [动手实践：在 Kubernetes 上部署模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8%20Kubernetes%20%E4%B8%8A%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B.md)
