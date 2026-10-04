# 安装 Docker

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/installing-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了在您的系统上运行 Docker，我们将进行安装。Docker 的用途是实现容器化，而镜像和容器是其主要概念。

安装 Docker 软件是构建镜像或运行容器的必要步骤，这包括我们计划为 Flask 预测服务创建的容器。

可以把 Docker 软件（常被称为 Docker 引擎）看作是管理镜像、容器、网络和存储卷的核心组件。如果没有安装它，您的计算机将无法识别 `Dockerfile`，也无法执行在隔离环境中打包和运行应用程序所需的命令。

### 为您的操作系统获取 Docker

安装 Docker 的具体步骤在很大程度上取决于您的计算机操作系统（如 Windows、macOS 或特定的 Linux 发行版），有时甚至还取决于处理器类型（如 Intel 或 Apple 芯片）。由于这些说明是平台专用的，并且 Docker 会频繁更新，本课程将不在此重复详细的安装步骤。

获取安装说明最好和最新的来源始终是官方 Docker 文档。

**访问 Docker 官方网站开始使用：** [获取 Docker](https://docs.docker.com/get-docker/)

查找与您的操作系统相关的部分：

- **Windows:** 您通常会安装 **Docker Desktop for Windows**。它提供图形界面以及命令行工具，并管理 Docker 引擎，通常使用适用于 Linux 2 的 Windows 子系统（WSL 2）后端以提升性能。请仔细遵循 Docker 网站上的说明，注意 Windows 版本和硬件虚拟化支持等系统要求。
- **macOS:** 您将安装 **Docker Desktop for Mac**。与 Windows 版本类似，它提供了一个方便的软件包，包含图形界面和必要的命令行工具。请为您的 Mac 处理器（Intel 芯片或 Apple 芯片）选择正确的下载。安装过程通常很直接，涉及下载 `.dmg` 文件并将 Docker 应用程序拖动到您的“应用程序”文件夹。
- **Linux:** 在 Linux 上的安装因您的发行版（例如 Ubuntu、Debian、Fedora、CentOS）而异。Docker 文档为每个主要发行版提供了具体说明。安装通常涉及设置 Docker 的软件包仓库，并使用您的发行版软件包管理器（例如 Ubuntu/Debian 的 `apt` 或 Fedora/CentOS 的 `dnf`/`yum`）安装 Docker 引擎软件包（`docker-ce`、`docker-ce-cli`、`containerd.io`、`docker-buildx-plugin`、`docker-compose-plugin`）。一些 Linux 发行版也可能提供 Docker Desktop 作为替代方案。

### 安装后步骤（尤其适用于 Linux）

安装 Docker 后，尤其是在 Linux 上，您可能需要执行一些额外配置。一个常见步骤是将您的用户帐户添加到 `docker` 组。这允许您直接运行 Docker 命令，而无需每次都以 `sudo` 为前缀。

```bash
# 在 Linux 上将当前用户添加到 docker 组的示例命令
# 您可能需要注销并重新登录才能使此更改生效。
sudo usermod -aG docker $USER
```

请参阅您的发行版在 Docker 官方文档中“Linux 安装后步骤”部分，以获取最准确的指导。

### 验证您的安装

安装过程完成后，最好验证 Docker 是否正确安装以及命令行界面 (CLI) 是否正在运行。打开您的终端或命令提示符，然后运行以下命令：

```bash
docker --version
```

如果 Docker 已正确安装，您应该会看到显示已安装 Docker 版本的输出，类似于这样（确切的版本号会有所不同）：

```
Docker version 25.0.3, build 4debf41
```

看到版本号可以确认您的系统识别 `docker` 命令，并且 Docker 引擎可能正在运行或已准备好启动。如果您遇到“command not found”（命令未找到）之类的错误消息，请仔细检查安装步骤，确保 Docker Desktop（如果适用）正在运行，并验证 Docker 的位置是否包含在您系统的 PATH 环境变量中。

Docker 在您的机器上成功安装后，您现在就拥有了开始打包应用程序的必要工具。在下一节中，我们将学习如何编写 `Dockerfile`，它是 Docker 用于构建镜像的蓝图。

## 参考资料

- [Get Docker](https://docs.docker.com/get-docker/) — Docker, Inc. (2024)
  Windows和macOS上安装Docker Desktop，以及在各种Linux发行版上安装Docker Engine的官方最新说明。
- [Docker Up and Running: Building High-Performance Microservices with Docker](https://www.oreilly.com/library/view/docker-up-and/9781492080838/) — Karl Matthias, Sean P. Kane (2023)
  Publisher: O'Reilly Media
  介绍Docker概念、架构和实际应用，为已安装的Docker Engine提供额外背景信息。
- [Docker Engine overview](https://docs.docker.com/engine/docker-overview/) — Docker, Inc. (2024)
  描述Docker Engine的基本组件和架构，包括镜像、容器及其管理。

---

[上一节](03-Docker%20%E6%A0%B8%E5%BF%83%E7%90%86%E5%BF%B5%EF%BC%9A%E9%95%9C%E5%83%8F%E4%B8%8E%E5%AE%B9%E5%99%A8.md) · [下一节](05-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Dockerfile.md)
