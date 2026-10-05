# Docker 应用打包介绍

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/intro-docker-packaging)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将一个在本地提供机器学习 (machine learning)预测能力的 FastAPI 应用从开发机器迁移到测试或生产服务器会带来很大的挑战。如何确保特定版本的 Python、所有必需的库（如 FastAPI、Uvicorn、scikit-learn、Pydantic）、系统依赖，甚至训练好的模型文件本身，在目标机器上配置完全一致是一个复杂的问题。操作系统、已安装软件包或配置的差异可能导致意想不到的错误和故障，这就是著名的“在我的机器上能运行”的问题。

这就是容器化，特别是使用 Docker，变得非常有用的地方。Docker 提供了一种方式，可以将你的应用连同其*所有*依赖项打包成一个名为容器的标准化单元。可以把容器想象成一个轻量、隔离的盒子，它包含了你的应用运行所需的一切：代码、运行时（如 Python）、系统工具、系统库、设置，以及在我们的案例中，序列化的机器学习模型。

与虚拟化整个操作系统的传统虚拟机（VM）不同，容器虚拟化的是操作系统*内核*。这意味着容器共享主机系统的内核，但拥有自己独立的进程空间、文件系统和网络接口。这使得它们比虚拟机轻量得多，启动也更快。

> 容器共享宿主机的操作系统内核，这使得它们比虚拟机更轻量，因为虚拟机每个都需要一个完整的客户机操作系统。

对于部署提供机器学习模型的 FastAPI 应用，Docker 提供多项优点：

- **环境一致性：** Docker 容器无论部署在哪里都能一致运行，无论是你的笔记本电脑、队友的机器、测试服务器还是云平台。这排除了与环境相关的错误。
- **依赖项打包：** Docker 将你的 FastAPI 代码、正确的 Python 版本、所有 `pip` 安装的库（在 `requirements.txt` 中指定）、任何系统级依赖，*以及*你训练好的模型文件（如 `.pkl` 或 `.joblib` 文件）打包成一个单一的、自包含的单元。
- **部署简化：** 替代复杂的设置脚本，部署通常变得像拉取 Docker 镜像并运行它一样简单。这简化了将机器学习模型投入生产的过程。
- **隔离性：** 你的 FastAPI 应用运行在自己独立的隔离环境中，避免与可能在同一宿主机上运行的其他应用或库发生冲突。

这个过程通常涉及三个主要组成部分：

1. **Dockerfile：** 一个文本文件，包含如何构建 Docker 镜像的指令。它指定了基础操作系统、安装依赖的命令、复制应用代码和模型文件的命令，以及容器启动时运行的命令。你将在下一节学习如何为 FastAPI 编写一个定制的 Dockerfile。
2. **Docker 镜像：** 从 Dockerfile 构建的只读模板。它包含应用及其所有依赖项。镜像可以存储并通过 Docker Hub 等注册表共享。
3. **Docker 容器：** Docker 镜像的一个可运行实例。你可以启动、停止和管理容器。每个容器都根据镜像定义作为独立的进程运行。

通过使用 Docker，你可以封装整个机器学习预测服务，使其可移植、可复现，并且在开发和部署生命周期的不同阶段管理起来更为便捷。以下章节将指导你为你的应用创建 `Dockerfile`、构建镜像，并将其作为容器运行。

## 参考资料

- [Docker overview](https://docs.docker.com/get-started/overview/) — Docker, Inc. (2024)
  提供Docker的基础知识、架构以及镜像和容器等核心组件。
- [Docker in Action, Second Edition](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff and Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本全面理解和应用Docker进行应用程序打包和部署的指南。
- [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115784/) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2020)
  Publisher: O'Reilly Media; Pages: 405
  讨论机器学习系统的部署模式，常使用容器化技术实现可复现性和可移植性。

---

[上一节](../05-%E5%BC%82%E6%AD%A5%E6%93%8D%E4%BD%9C%E4%B8%8E%E6%80%A7%E8%83%BD/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%BC%82%E6%AD%A5%E6%93%8D%E4%BD%9C.md) · [下一节](02-%E7%BC%96%E5%86%99%20FastAPI%20%E5%BA%94%E7%94%A8%E7%9A%84%20Dockerfile.md)
