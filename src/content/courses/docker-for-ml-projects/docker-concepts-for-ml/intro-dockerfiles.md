---
course: "docker-for-ml-projects"
chapter: "docker-concepts-for-ml"
lesson: "intro-dockerfiles"
sourceId: 5236
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml/intro-dockerfiles"
title: "Dockerfile 简介"
description: "Dockerfile 语法概述及其在构建自定义镜像中的作用。"
order: 4
plots: []
sourceHash: "f57f4ccc0371f789e39152e6a239cfdbe97cbee9489c9ef73905320793daa9f8"
sourceCorrections: []
---

尽管从 Docker Hub 等注册中心获取预构建的 Docker 镜像是一个方便的起点，但您通常需要创建专门根据机器学习 (machine learning)项目需求定制的环境。您可能需要精确的 Python 版本、特定的机器学习库（如 TensorFlow、PyTorch 或 Scikit-learn）、系统依赖项，或者包含您自己的项目代码。这时就需要用到 Dockerfile。

Dockerfile 是一个文本文档，其中包含 Docker 用于逐步构建自定义容器镜像的一系列指令。可以将其看作是创建环境的食谱或蓝图。通过编写 Dockerfile，您可以定义运行应用程序所需的一切：操作系统基础、系统工具、语言运行时、代码依赖项、您的应用程序代码，以及应用程序如何启动。

这种自动化对机器学习中的可复现性非常重要。如果您与他人分享 Dockerfile 以及您的代码，其他人就可以构建出*完全*相同的镜像，保证环境在开发、测试和生产系统之间保持一致。这解决了由于库版本或系统配置的微小差异而经常出现的“在我的机器上能运行”的问题。

### 基本结构和指令

Dockerfile 由一系列指令组成，这些指令通常以大写字母书写，后跟参数 (parameter)。Docker 从上到下读取这些指令，逐一执行以形成最终镜像的层。以下是您会遇到的一些最常用指令：

- `FROM`: 每个 Dockerfile 都必须以 `FROM` 指令开始。它指定了您将要构建所基于的*基础镜像*。这可以是一个最小操作系统（如 `ubuntu:22.04`）、一个特定语言的镜像（如 `python:3.9-slim`），甚至是包含机器学习 (machine learning)框架或 CUDA 驱动的专用镜像（如 `nvidia/cuda:11.8.0-base-ubuntu22.04` 或 `tensorflow/tensorflow:latest-gpu`）。选择合适的基础镜像是一个重要考量，我们之后会讲到。

  ```dockerfile
  FROM python:3.9-slim
  ```
- `RUN`: 此指令在当前镜像之上执行新层中的命令。它常用于安装系统软件包（使用 `apt-get` 或 `yum`）或 Python 库（使用 `pip` 或 `conda`）。多个 `RUN` 指令会创建多个层，但您可以使用 `&&` 链接命令，以便更好地管理层。

  ```dockerfile
  RUN apt-get update && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/*
  RUN pip install --no-cache-dir scikit-learn==1.2.0 pandas==1.5.2
  ```
- `COPY`: 此指令将文件或目录从您的本地机器（*构建上下文 (context)*）复制到容器镜像的文件系统中。您将使用此指令将您的 Python 脚本、需求文件（`requirements.txt`）、配置文件甚至预训练 (pre-training)模型文件添加到镜像中。

  ```dockerfile
  COPY ./requirements.txt /app/requirements.txt
  COPY ./src /app/src
  ```
- `WORKDIR`: 设置后续所有 `RUN`、`CMD`、`ENTRYPOINT`、`COPY` 和 `ADD` 指令的工作目录。如果目录不存在，它将被创建。这有助于整理容器的文件系统。

  ```dockerfile
  WORKDIR /app
  ```
- `CMD`: 指定从镜像启动容器时要执行的默认命令。Dockerfile 中只能有一个 `CMD` 指令。如果您在运行容器时提供命令（`docker run my-image <command>`），它将覆盖默认的 `CMD`。它常用于启动应用程序的主进程，例如运行 Python 脚本。

  ```dockerfile
  CMD ["python", "src/train.py"]
  ```
- `ENTRYPOINT`: 也配置一个容器，使其以可执行文件的形式运行。它与 `CMD` 相似，但通过 `docker run` 传递的命令将作为参数附加到 `ENTRYPOINT` 可执行文件。这对于创建像特定工具一样运行的镜像很有用。
- `EXPOSE`: 通知 Docker 容器在运行时监听指定的网络端口。它实际上不会发布端口；其作用是作为构建镜像者和运行容器者之间的文档。

  ```dockerfile
  EXPOSE 8000
  ```

Dockerfile 中的注释以 `#` 开头。

### 一个简单示例

这是一个最简单的 Dockerfile，用于设置一个带有 Python 3.9 的环境，并安装 Scikit-learn 和 Pandas：

```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 在容器中设置工作目录
WORKDIR /app

# 将 requirements 文件复制到容器中的 /app 目录
COPY requirements.txt .

# 安装 requirements.txt 中指定的任何所需软件包
# 使用 --no-cache-dir 以减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 复制其余的应用程序代码
COPY . .

# 指定容器启动时运行的命令
CMD ["python", "your_script.py"]
```

要从这个 Dockerfile 构建镜像（假设它在当前目录中保存为 `Dockerfile`），您通常会运行如下命令：

```bash
docker build -t my-ml-app .
```

`-t my-ml-app` 标签为您的镜像指定一个易于识别的名称，而 `.` 表示构建上下文 (context)（`COPY` 指令可用的文件集）是当前目录。Docker 逐行执行指令，并在可能的情况下缓存结果，以创建您的最终镜像 `my-ml-app`。

了解 Dockerfile 对于在机器学习 (machine learning)中应用 Docker 的优势十分重要。它们使您能够精确地定义、版本化和共享运行训练脚本或服务模型所需的环境，保证一致性并简化协作和部署。在下一章中，我们将讨论如何编写更复杂的 Dockerfile，特别是用于构建机器学习环境的 Dockerfile。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  Publisher: Docker
  Dockerfile语法、指令和最佳实践的官方详细文档。
- [Docker: Up & Running](https://www.oreilly.com/library/view/docker-up-running-3rd/9781098131821/) — Sean P. Kane, Karl Matthias (2023)
  Publisher: O'Reilly Media
  涵盖Docker基本概念的指南，包括如何为各种应用编写有效的Dockerfile。
