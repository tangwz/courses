---
course: "planning-optimizing-ai-infrastructure"
chapter: "containerization-orchestration-for-ml"
lesson: "docker-for-reproducible-environments"
sourceId: 7008
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/docker-for-reproducible-environments"
title: "Docker 在可复现环境中的使用介绍"
description: "理解如何使用 Docker 容器打包机器学习代码及其依赖项，以便在不同系统上一致运行。"
order: 1
plots: []
sourceHash: "572ee35630ace91cf30ee2545831255c1dc64e7284b4fad9e3c0a9fe0ed747ca"
sourceCorrections: []
---

将机器学习 (machine learning)项目从本地开发机转移到生产服务器时，经常会遇到令人沮丧的“在我的机器上能跑”的问题。一个在您的笔记本上运行完美的模型，可能因为 Python 版本的细微差异、依赖冲突或不兼容的系统库而在部署时失败。这种不一致性使得协作变得困难，部署也不可靠。

Docker 提供了一个有效的办法来解决这个问题，它引入了一种标准方式，将应用程序打包并在称为容器的隔离环境中运行。容器将您的应用程序代码及其所有必要的依赖项、库和配置文件捆绑在一起。这样，无论是在开发者的笔记本电脑、本地服务器还是云虚拟机上，只要安装了 Docker，这个打包好的程序就能统一且一致地运行。

### 容器与虚拟机对比

区分容器和虚拟机（VM）很有帮助，因为它们解决的是类似问题，但方法不同。虚拟机模拟的是一个完整的计算机系统，包含一个完整的客户操作系统运行在宿主操作系统之上。这提供了强大的隔离性，但代价是会产生大量的开销，包括大小、启动时间和资源消耗。

相比之下，容器则更轻量。它们虚拟化操作系统本身，允许多个容器运行在单个宿主机上并共享宿主机的操作系统内核。它们只打包应用程序代码及其特定的依赖项。这种高效性意味着在给定服务器上可以运行比虚拟机多得多的容器，而且它们可以几乎立即启动。

> 容器通过容器引擎共享宿主机的操作系统内核，这使得它们比虚拟机更轻量、速度更快。虚拟机则需要为每个应用程序提供一个完整的客户操作系统。

### Docker 的核心组件

使用 Docker 涉及一些您会经常用到的核心组件：

- **Dockerfile：** 这是一个简单的文本文件，包含一系列关于如何构建 Docker 镜像的指令。它就像是您容器化环境的食谱或蓝图。您需要指定一个基础镜像（例如，官方 Python 或 NVIDIA CUDA 镜像），列出要安装的系统软件包，复制您的应用程序代码，并定义容器启动时要运行的命令。
- **镜像 (Image)：** 镜像是一个只读的静态模板，根据 Dockerfile 中的指令创建。它包含应用程序及其所有依赖项。镜像存储在注册中心（如 Docker Hub 或私有云注册中心），并用于创建运行中的容器。因为镜像分层构建，所以它们在存储和分发方面很高效。
- **容器 (Container)：** 容器是 Docker 镜像的一个可运行的实时实例。您可以创建、启动、停止和删除容器。每个容器都是运行在宿主机内核上的一个隔离进程，但它拥有自己的私有文件系统、网络和进程空间，所有这些都由创建它的镜像提供。

### Docker 对机器学习 (machine learning)的重要性

对于机器学习工作流程而言，容器化的好处非常突出。虽然像 `venv` 或 `conda` 这样的 Python 虚拟环境可以管理 Python 包依赖项，但它们无法创建真正可复现的环境。它们没有考虑系统级依赖项、环境变量或特定的 GPU 驱动版本，而所有这些都可能影响模型的行为。

Docker 直接解决了这些不足：

- **完整的依赖项封装：** `Dockerfile` 可以包含运行您的代码所需的一切。这不仅包括 `requirements.txt` 文件中的 Python 包，还包括通过 `apt-get` 安装的系统库、深度学习 (deep learning)框架所需的特定版本 CUDA，以及任何必要的环境变量。
- **保证可复现性：** 通过打包整个环境，您可以保证训练脚本或模型服务应用程序在任何地方都能完全相同地运行。这对于复现实验结果、调试以及确保训练和生产推理 (inference)之间的一致性非常有价值。
- **简化协作和部署：** 您可以共享 Docker 镜像，而无需共享代码和一长串设置说明。同事或 CI/CD 流水线只需运行该镜像，无需手动配置环境，这大幅简化了将机器学习应用程序从开发环境迁移到生产环境的过程。

采用 Docker，您可以为您的机器学习系统建立一个稳定且可预测的环境。这使您能够专注于构建和训练模型，确信底层环境是一致且可移植的。在下一节中，我们将通过为机器学习应用程序编写第一个 `Dockerfile` 来将此付诸实践。

## 参考资料

- [Docker overview](https://docs.docker.com/get-started/overview/) — Docker Docs (2023)
  Publisher: Docker
  提供了对 Docker 的基本理解，包括其架构、镜像和容器等核心组件，以及与虚拟机的区别。
- [Containers for reproducible research: A review](https://f1000research.com/articles/6-634) — Carl Boettiger, Dirk Eddelbuettel (2017)
  Journal: F1000Research; Publisher: F1000Research; Volume: 6; Pages: 634; DOI: [10.12688/f1000research.11472.2](https://doi.org/10.12688/f1000research.11472.2)
  全面回顾了容器化（特别是 Docker）如何解决科学计算和研究中可重复性面临的挑战。
- [Engineering MLOps: A Guide to Building High-Quality Machine Learning Systems](https://www.packtpub.com/product/engineering-mlops/9781801813133) — Emmanuel Raj, Mark Trevena (2022)
  Publisher: Packt Publishing
  讨论了 MLOps 的最佳实践，包括容器化（Docker）在创建可复现和可部署机器学习工作流中的作用。
