---
course: "docker-for-ml-projects"
chapter: "building-ml-dockerfiles"
lesson: "choosing-base-image"
sourceId: 5248
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/choosing-base-image"
title: "选择恰当的起始映像"
description: "比较适合机器学习工作负载的不同起始映像（例如，官方Python映像、CUDA映像）。"
order: 2
plots: []
sourceHash: "d063883e3b860834e67be6fceb98e674a85fce4f994cd9bc80bb6c4a5f23c59c"
sourceCorrections: []
---

每个 `Dockerfile` 都以 `FROM` 指令开头，指定了*起始映像*。可以将其看作是构建自定义环境的根基。这不仅仅是一种形式；您选择的起始映像会显著影响最终映像的大小、构建时间、安全状态以及机器学习 (machine learning)应用的兼容性。在此做出明智选择，为更流畅的开发和部署流程做好准备。

### 影响起始映像选择的因素

为您的机器学习 (machine learning)项目选择起始映像时，请考量以下因素：

1. **操作系统和核心库：** 您是否需要特定的Linux发行版（如Ubuntu或Debian）来获取某些系统包，或者一个最小化映像就足够？此处一个重要考量是C标准库：大多数发行版使用 `glibc`，而Alpine Linux使用 `musl libc`。这会影响与预编译Python wheels的兼容性，有时需要从源代码编译包，从而增加构建时间和复杂性。
2. **预装软件：** 您是否需要一个预装Python的映像？或者一个已经包含NVIDIA驱动、CUDA工具包和cuDNN用于GPU加速的映像？您是否需要特定的版本组合？
3. **映像大小：** 通常更受欢迎的是较小的映像。它们占用更少的磁盘空间，从镜像仓库拉取速度更快，并能减少部署时间。最小化映像通常具有更小的攻击面，可能提高安全性。然而，最小化有时会以牺牲便利性为代价，需要在您的 `Dockerfile` 中进行更多设置步骤。
4. **维护和信任：** 官方映像（如 `python`、`ubuntu`、`nvidia/cuda`）通常维护良好，并定期更新以解决安全漏洞。强烈建议使用官方或受信任的来源。

### 机器学习 (machine learning)的常见起始映像类别

我们来看看机器学习 `Dockerfile` 的一些常见起点：

#### 1. 官方Python映像

如果您的主要需求是特定Python版本，这些通常是一个实际的起点。

- `python:<version>`（例如，`python:3.10`）：基于最新稳定版Debian。包含Python和 `pip`，以及构建许多Python扩展所需的常用操作系统库。这是一个很好的通用选择，但相对较大。
- `python:<version>-slim`（例如，`python:3.10-slim`）：一个更小的变体，也基于Debian。它移除了不常用操作系统开发库。如果您的Python包需要编译，您可能需要手动安装一些系统包（`build-essential`，特定库头文件如 `libpq-dev`）。这通常在大小和便利性之间提供了良好平衡。
- `python:<version>-alpine`（例如，`python:3.10-alpine`）：基于非常小巧的Alpine Linux发行版。导致映像尺寸显著减小。然而，Alpine使用 `musl libc` 而不是更常见的 `glibc`。这意味着许多预编译Python wheels（`manylinux`）无法直接运行，可能需要在Docker构建过程中进行耗时的编译，或者寻找Alpine专用的包。请谨慎使用，特别是当您有复杂的二进制依赖时。

#### 2. 操作系统起始映像

有时，您可能需要更多控制或Python映像中不易获得的特定系统工具。

- `ubuntu:<version>`（例如，`ubuntu:22.04`）
- `debian:<version>`（例如，`debian:bullseye-slim`）

从一个纯净的操作系统映像开始意味着您需要自己安装Python、`pip` 以及所有其他依赖。这提供了最大程度的控制，但需要更多 `Dockerfile` 指令。通常仅在您有复杂的非Python系统依赖或非常具体的操作系统配置要求时有必要。

#### 3. NVIDIA CUDA映像

如果您的机器学习工作负载需要GPU加速，使用NVIDIA的官方CUDA映像是标准方法。这些映像预配置了必要的NVIDIA驱动、CUDA工具包和cuDNN库，它们兼容在支持GPU的主机上运行的Docker环境（使用NVIDIA容器工具包）。

- `nvidia/cuda:<cuda_version>-base-<os_version>`（例如，`nvidia/cuda:11.8.0-base-ubuntu22.04`）：仅包含CUDA运行时库。如果您*不需要*容器内部的CUDA编译器（`nvcc`）或开发库（例如，仅用于推理 (inference)），则适用。
- `nvidia/cuda:<cuda_version>-runtime-<os_version>`（例如，`nvidia/cuda:11.8.0-runtime-ubuntu22.04`）：包含基本运行时库，以及CUDA数学库和NCCL。通常足以运行预编译应用或捆绑自身开发头文件（如果需要）的Python框架。
- `nvidia/cuda:<cuda_version>-devel-<os_version>`（例如，`nvidia/cuda:11.8.0-devel-ubuntu22.04`）：包含完整的CUDA工具包，包括编译器（`nvcc`）、调试工具、头文件和静态库。如果您需要在容器内部编译CUDA代码或某些Python库（如从源代码构建PyTorch），则必要。这将产生最大的映像。

使用这些映像时，您通常会在其之上安装Python和您的机器学习库。务必查阅NVIDIA文档和您的机器学习框架要求，以获取推荐的CUDA/cuDNN版本。

#### 4. 机器学习框架官方映像

主要机器学习框架通常提供自己的官方Docker映像，这些映像基于合适的起始映像（如Ubuntu + CUDA）构建。

- `tensorflow/tensorflow:<version>`（例如，`tensorflow/tensorflow:2.11.0-gpu`）
- `pytorch/pytorch:<version>`（例如，`pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime`）

**优点：**

- 便利性：预装了框架、特定CUDA/cuDNN版本，通常还有常用依赖。
- 兼容性：版本经过测试，可协同工作。

**缺点：**

- 大小：由于捆绑了许多组件，它们可能非常庞大。
- 较少控制：您可能获得不需要的库版本（CUDA，Python）或操作系统映像。
- 潜在更新较慢：可能略滞后于官方Python或CUDA映像更新。

这些映像非常适合快速起步，或确保特定复杂环境（如特定PyTorch构建与特定CUDA版本）在无需手动配置的情况下正确设置。

### 做出选择：一个决策过程

选择正确的起始映像涉及平衡这些因素。以下是一个简化的决策流程：

> 一个用于为机器学习 (machine learning)项目选择合适的Docker起始映像的决策指南。

对于使用Scikit-learn、Pandas等库，以及可能使用TensorFlow或PyTorch（特别是通过pip安装时）的许多标准机器学习任务，以 `python:<version>-slim` 作为起点通常是一个明智的默认选项。如果需要GPU支持，则转而使用官方 `nvidia/cuda` 映像或框架专用的GPU映像。如果需要极致的尺寸优化，并且您能处理潜在的兼容性障碍，那么 `python:<version>-alpine` 是一个选项。

周全地选择您的起始映像是构建高效、可复现和可维护的Docker化机器学习环境的第一步。它为管理依赖、复制代码和配置运行时设定了情境，这些将在后续章节中介绍。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  Publisher: Docker
  Dockerfile指令的官方指南，包含`FROM`指令和高效镜像构建的考量。
- [python - Official Images](https://hub.docker.com/_/python) — Docker Hub (2024)
  提供官方Python Docker镜像的详细信息，涵盖标准、slim和Alpine变体，以及它们底层libc库的区别（glibc与musl libc）。
- [nvidia/cuda - Official Images](https://hub.docker.com/r/nvidia/cuda) — NVIDIA (2024)
  NVIDIA CUDA Docker镜像的官方信息，区分了用于GPU加速的`base`、`runtime`和`devel`变体。
- [Docker: Up & Running: Building Scalable and Secure Cloud Applications](https://www.oreilly.com/library/view/docker-up-running/9781492095360/) — Karl Matthias, Sean P. Kane (2022)
  Publisher: O'Reilly Media
  一本全面的Docker指南，提供Dockerfile结构、镜像优化和影响基础镜像选择的安全方面的实用建议。
- [MLOps: Continuous Delivery and Automation in Machine Learning](https://www.oreilly.com/library/view/mlops-continuous-delivery-and/9781492083299/) — Vishnu Lakshmanan, Vignesh Sukumaran (2021)
  Publisher: O'Reilly Media
  提供将Docker集成到ML工作流中的背景，强调基础镜像决策如何影响MLOps环境中的可复现性和部署。
