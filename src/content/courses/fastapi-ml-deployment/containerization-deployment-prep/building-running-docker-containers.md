---
course: "fastapi-ml-deployment"
chapter: "containerization-deployment-prep"
lesson: "building-running-docker-containers"
sourceId: 5347
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/building-running-docker-containers"
title: "构建和运行 Docker 容器"
description: "用于构建 Docker 镜像和从这些镜像运行容器的命令。"
order: 4
plots: []
sourceHash: "1b524c3daacb301438cfb83e09ef8c5038fc3cec7e931706f2bf3e5e26f2becd"
sourceCorrections: []
---

一个 `Dockerfile` 充当着容器化 FastAPI 应用的蓝图。利用这份蓝图，可以构建一个实际的镜像，然后基于该镜像启动一个容器。这个过程涉及两个主要的 Docker 命令：`docker build` 和 `docker run`。

### 使用 `docker build` 构建 Docker 镜像

`docker build` 命令用于从 `Dockerfile` 和一个“上下文 (context)”创建 Docker 镜像。上下文是指位于指定 `PATH` 或 `URL` 中的文件集合。这些文件在构建过程中会发送给 Docker 守护进程。

你最常用的语法是：

```bash
docker build -t <image_name>:<tag> <path_to_build_context>
```

我们来分解一下各部分：

- `docker build`：命令本身。
- `-t <image_name>:<tag>`：这个选项是 `--tag` 的缩写，非常重要。它允许你为镜像指定一个易读的名称和标签。格式通常是 `repository_name:tag`。
  - `repository_name`：通常反映应用程序的名称，例如 `fastapi-ml-api`。
  - `tag`：表示特定版本或变体，例如 `v1.0`、`latest` 或 Git 提交哈希。使用 `latest` 很常见，但在生产环境中应谨慎使用，因为指定具体版本更有利于复现。
  - 示例：`-t fastapi-ml-api:latest` 或 `-t mycompany/prediction-service:0.1.0`。使用标签会使你之后管理和引用镜像时更加方便。
- `<path_to_build_context>`：这指定了你的 `Dockerfile` 和它需要的应用程序文件（例如你的 Python 代码、`requirements.txt` 以及如果直接复制的话可能还有模型文件）的位置。最常见的情况是，你会从项目根目录（`Dockerfile` 所在的目录）运行此命令，因此路径只需是 `.`（代表当前目录）。

当你执行 `docker build` 时，Docker 守护进程会执行以下操作：

1. 将构建上下文（指定路径中的所有文件）发送给守护进程。请注意构建上下文中不需要的大文件；使用 `.dockerignore` 文件（类似于 `.gitignore`）来排除它们，以加快构建速度。
2. 读取 `Dockerfile`。
3. 按顺序执行 `Dockerfile` 中的每条指令。每条指令通常会在镜像中创建一个新的层。
4. 使用构建缓存：如果一条指令及其输入文件自上次构建以来没有改变，Docker 会重用缓存中已有的层，从而显著加快后续构建的速度。这体现了在 `Dockerfile` 中有效组织指令顺序的重要性（例如，在复制更常变化的应用程序代码之前，先复制并安装依赖项）。
5. 成功完成后，输出新构建镜像的 ID。如果你使用了 `-t` 标志，它还会将你选择的名称和标签与此镜像 ID 关联起来。

**示例：**

假设你的 `Dockerfile` 位于当前目录，并且与你的应用程序代码（`main.py`、`requirements.txt`、模型文件等）在一起，你会这样构建镜像：

```bash
docker build -t fastapi-ml-api:latest .
```

你会看到输出详细说明构建过程的每一步。完成后，你可以使用 `docker images` 列出本地镜像，看到带有 `latest` 标签的 `fastapi-ml-api`。

### 使用 `docker run` 运行 Docker 容器

现在你已经有了包含你的 FastAPI 应用程序、其依赖项和 ML 模型的 Docker 镜像，你可以使用 `docker run` 命令从它创建并启动一个容器。

基本语法是：

```bash
docker run [OPTIONS] <image_name>:<tag>
```

以下是一些你经常会用到的选项（`[OPTIONS]`）：

- `-p <host_port>:<container_port>`：这是端口映射选项，是 `--publish` 的缩写。它将你的宿主机上的一个端口连接到容器内部的一个端口。
  - `container_port`：你的应用程序在容器内部监听的端口（例如，在你的 `CMD` 指令中指定的端口，如 `uvicorn main:app --host 0.0.0.0 --port 80`）。在容器内部使用 `0.0.0.0` 作为主机地址，可以使其从容器的网络命名空间外部访问。
  - `host_port`：你将用于从宿主机的浏览器或诸如 `curl` 等工具访问应用程序的端口。
  - 示例：`-p 8000:80` 将宿主机上的 8000 端口映射到容器内部的 80 端口。然后你就可以通过 `http://localhost:8000` 访问你的 API 了。
- `-d`：这会以分离模式运行容器，意味着它在后台运行，你的终端会被释放。如果没有 `-d`，你的终端将连接到容器的输出（日志）。
- `--name <container_name>`：为你的运行中容器指定一个名称。如果你不提供名称，Docker 会生成一个随机名称。为容器命名会使它们更容易管理（例如，停止、删除、检查）。示例：`--name ml_api_instance_1`。
- `-e <VARIABLE_NAME>=<value>`：在容器内部设置环境变量。这是向应用程序传递配置设置的常用方式，下一节会提到。示例：`-e MODEL_PATH=/app/models/model.pkl`。
- `--rm`：容器退出时自动删除。适用于临时容器或在开发过程中使用，以避免系统被停止的容器占用空间。

**示例：**

要基于我们之前构建的镜像运行一个容器，将宿主机的 8000 端口映射到容器的 80 端口（假设你的 `Dockerfile` 的 `CMD` 在 80 端口运行 Uvicorn），并以分离模式运行：

```bash
docker run -d -p 8000:80 --name my_api fastapi-ml-api:latest
```

运行此命令后，Docker 将输出新启动容器的唯一 ID。

### 检查和管理容器

一旦你的容器开始运行（特别是在分离模式下），你可能会想与其进行交互：

- **检查运行中的容器**：使用 `docker ps`。这会列出所有当前运行的容器，显示它们的 ID、名称、镜像来源、状态、端口等。添加 `-a` 标志（`docker ps -a`）可以查看所有容器，包括已停止的容器。
- **查看日志**：如果以分离模式运行，你可以使用 `docker logs <container_name_or_id>` 查看应用程序日志（Uvicorn 的标准输出/标准错误）。使用 `-f` 标志（`docker logs -f <container_name_or_id>`）可以实时跟踪日志。
- **停止容器**：使用 `docker stop <container_name_or_id>`。
- **启动已停止的容器**：使用 `docker start <container_name_or_id>`。
- **删除容器**：使用 `docker rm <container_name_or_id>`。通常你需要先停止容器才能删除它，除非你使用强制标志 `-f`。使用 `docker run --rm` 可以避免手动执行此操作。
- **访问 API**：通过上面的示例 `docker run` 命令，你应该能够从宿主机通过 `http://localhost:8000` 访问你的 API 端点（例如，通过 `http://localhost:8000/docs` 访问 Swagger UI）。

使用 `docker build` 和 `docker run` 构成了将你的应用程序代码和 `Dockerfile` 转换为运行中的、隔离的服务的核心工作流程。通过构建镜像，你将应用程序及其所有依赖项打包在一起，包括特定的 Python 解释器和 ML 模型本身。将此镜像作为容器运行可提供一致的环境，确保无论容器部署在哪里，你的应用程序行为都一致。

## 参考资料

- [docker build](https://docs.docker.com/reference/cli/docker/build/) — Docker, Inc. (2024)
  `docker build` 命令的官方文档，详细说明了其语法、选项以及如何从 Dockerfile 和构建上下文创建 Docker 镜像。
- [docker run](https://docs.docker.com/reference/cli/docker/run/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  `docker run` 命令的官方文档，描述了其语法、端口映射、分离模式、命名和环境变量等选项，用于启动容器。
- [The Docker Deep Dive](https://www.amazon.com/Docker-Deep-Dive-Nigel-Poulton/dp/0138247070) — Nigel Poulton (2023)
  Publisher: Pearson
  一份全面的指南，讲解 Docker 基础知识，包括镜像、容器、Dockerfile 以及构建/运行工作流程，提供概念和实践细节。
