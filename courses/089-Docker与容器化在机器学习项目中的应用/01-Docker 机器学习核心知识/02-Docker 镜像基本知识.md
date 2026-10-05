# Docker 镜像基本知识

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml/docker-image-fundamentals)

[返回章节目录](README.md) · [返回课程目录](../README.md)

容器化提供了一种有效方法来打包和隔离机器学习 (machine learning)应用程序，解决了依赖冲突等常见问题，并确保了不同环境之间的一致性。实现这一点的核心组成部分是 Docker 镜像。可以将 Docker 镜像看作是创建这些标准化环境的蓝图或模板。

### 什么是 Docker 镜像？

Docker 镜像是一个轻量、独立、可执行的软件包，包含运行软件所需的一切：代码、运行时（如 Python）、系统工具、系统库和配置。它是一个只读模板。当您想运行镜像定义的软件时，会创建一个 *容器*，它本质上是该镜像的一个运行实例。

如果您熟悉面向对象编程，镜像就像一个类，而容器就像一个对象（类的实例）。或者，与虚拟化技术类比，镜像有点像虚拟机模板或快照，但它更轻量，因为它共享主机操作系统的内核，而不是打包一个完整的操作系统。

对于机器学习 (machine learning)项目，镜像通常会包含：

- 您的 Python 训练或推理 (inference)脚本。
- 特定的 Python 解释器版本（例如 Python 3.9）。
- 所有必需的库（例如 `tensorflow`、`pytorch`、`scikit-learn`、`pandas`、`numpy`），并锁定到特定版本。
- 任何必要的系统级依赖或工具。
- 您的应用程序所需的配置文件或环境变量定义。

### 分层文件系统

Docker 镜像不是一个整体的块。它们由一系列只读层构建而成，层层堆叠。每个层代表镜像 Dockerfile 中的一个指令，Dockerfile 是用于构建镜像的脚本。例如，安装系统包、复制源代码或设置环境变量通常会创建一个新层。

> 镜像层的简化视图。Dockerfile 中的每个指令都会添加一个新层。当容器运行时，会在顶部添加一个薄的可写层。

这种分层架构，通常通过 OverlayFS 等技术实现，具有多项优点，尤其与机器学习 (machine learning)相关：

1. **效率：** Docker 会缓存层。如果您修改 Dockerfile，Docker 只需重新构建已更改的层及其后续层。如果您的基础操作系统和大型机器学习库层未更改，那么重新构建镜像以包含更新的应用程序代码会快得多。
2. **共享：** 不同的镜像可以共享公共层。如果您有多个基于相同 Python 版本或相同 CUDA 基础镜像的机器学习镜像，该层在您的磁盘上只存储一次，并在它们之间共享。这在拉取镜像时节省了磁盘空间和网络带宽。
3. **大小管理：** 像 PyTorch 或 TensorFlow 这样的机器学习库，特别是支持 GPU 的库，可能非常大。了解层是优化镜像大小的第一步，我们稍后将使用多阶段构建等技术来介绍这个话题。

### 基础镜像

每个 Docker 镜像都从一个 *基础镜像* 开始。这在 Dockerfile 中通过 `FROM` 指令指定。基础镜像为您的环境提供了底层。机器学习 (machine learning)项目常用选项包括：

- **官方 Python 镜像：** `python:3.10-slim-bullseye` 提供了一个基于 Debian 的最小环境，预装了特定 Python 版本。适用于一般 Python 应用程序。
- **操作系统镜像：** `ubuntu:22.04` 或 `debian:bullseye` 提供更完整的 Linux 环境，如果您需要精简版 Python 镜像中未包含的特定系统工具或库。
- **供应商机器学习镜像：** 像 NVIDIA 这样的公司提供了为 GPU 使用优化的基础镜像（`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`），其中包含必要的 CUDA 驱动和库。特定框架的镜像（`tensorflow/tensorflow:latest-gpu` 或 `pytorch/pytorch:latest`）在此基础上构建，添加机器学习框架本身。

选择正确的基础镜像需要在大小、包含的组件以及与您需求的兼容性之间取得平衡。我们将在下一章中更详细地介绍这个选择过程。

### 镜像与容器的对比

明确区分镜像和容器是很重要的：

- **镜像：** 包含应用程序及其依赖的只读模板。它独立于任何运行中的进程而存在。
- **容器：** 镜像的一个可运行实例。当您执行 `docker run <image_name>` 时，Docker 会使用该镜像创建一个容器。

主要区别在于，当容器创建时，会在只读镜像层之上添加一个薄的 *可写层*。容器运行期间进行的任何更改，例如写入日志文件、创建临时数据或修改容器文件系统内的配置，都发生在此可写层中。如果容器被删除，此可写层及其中的任何更改都将丢失，除非您明确使用 Docker 卷或绑定挂载等机制将数据持久化到容器生命周期之外，我们将在第 3 章中讨论这些机制。

### 不变性：实现可重复性的方法

Docker 镜像是不可变的。一旦镜像构建完成，就不能更改。如果您需要更新代码、更改依赖项或修改配置，您必须：

1. 修改源文件（例如，您的 Dockerfile 或应用程序代码）。
2. 从修改后的源文件构建一个 *新* 镜像。这个新镜像将拥有不同的 ID。

这种不变性是机器学习 (machine learning)中实现可重复性的一个重要前提。镜像捕获了构建时环境、依赖项和代码的 *确切* 状态。如果您共享此镜像（例如，通过 Docker Hub 等注册表），其他人可以拉取并运行容器，确保他们拥有与您使用的完全相同的环境。这消除了在协作或部署机器学习模型时常遇到的“在我的机器上能跑”问题。

理解 Docker 镜像、层、基础镜像以及镜像与容器关系的这些要点，为有效构建、管理和共享机器学习工作流所需的一致环境提供了依据。

## 参考资料

- [Docker overview](https://docs.docker.com/get-started/overview/) — Docker, Inc. (2025)
  Publisher: Docker, Inc.
  介绍 Docker 核心概念，例如镜像、容器、分层架构及其相互关系，作为基础技术参考。
- [Docker Deep Dive: Zero to Docker Hero](https://leanpub.com/dockerdeepdive) — Nigel Poulton (2025)
  Publisher: Self-published (via Leanpub); Pages: 280
  一本全面的指南，深入解释 Docker 镜像、Dockerfile、分层文件系统和容器构建过程。
- [Practical MLOps: Operationalizing Machine Learning Models](https://books.google.com/books/about/Practical_MLOps_Operationalizing_Machin.html?id=lBwxEAAAQBAJ) — Noah Gift, Alfredo Deza (2021)
  Publisher: O'Reilly Media; Pages: 439
  涵盖容器化，尤其是 Docker 镜像在构建可复现和可扩展的机器学习工作流以及部署 ML 模型方面的应用。

---

[上一节](01-%E4%B8%BA%E4%BD%95%E5%AE%B9%E5%99%A8%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE%EF%BC%9F.md) · [下一节](03-%E5%AE%B9%E5%99%A8%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E7%AE%A1%E7%90%86.md)
