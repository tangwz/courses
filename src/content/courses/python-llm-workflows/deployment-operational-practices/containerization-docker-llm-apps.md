---
course: "python-llm-workflows"
chapter: "deployment-operational-practices"
lesson: "containerization-docker-llm-apps"
sourceId: 5220
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-10-deployment-operational-practices/containerization-docker-llm-apps"
title: "使用 Docker 进行容器化"
description: "使用 Docker 为您的 LLM 应用创建可移植、可重现的运行环境。"
order: 2
plots: []
sourceHash: "f279454e367866275209bcc239efe7af89d7b15e734357b56ac744be297b410d"
sourceCorrections: []
---

在部署 LLM 应用代码时，一个主要挑战是确保其在不同环境中能够一致运行。开发机器、测试服务器以及最终的生产环境，可能在操作系统、安装的库和系统配置方面存在细微（或明显）的差异。这些不一致会导致令人头疼的“在我的机器上能运行”问题，引起部署失败和操作上的困扰。容器化，特别是使用 Docker，通过将应用及其依赖项打包成一个标准化、可移植的单元，为这个难题提供了一个有效的解决方法。

### 什么是 Docker 容器？

可以将 Docker 容器看作一个轻量、独立、可执行的包，其中包含运行一段软件所需的一切：代码、运行时（如 Python 解释器）、系统工具、系统库和设置。与虚拟化整个操作系统的传统虚拟机（VM）不同，容器虚拟化的是操作系统*内核*。这意味着容器共享宿主系统的内核，但拥有自己独立的进程空间、文件系统和网络接口。这种做法使它们比虚拟机轻便得多，启动速度也更快。

创建容器的蓝图称为 **Docker 镜像**。镜像是一个只读模板，包含应用及其依赖项。您可以使用在一个名为 **Dockerfile** 的特殊文件中定义的指令来构建镜像。当您运行一个镜像时，您会创建它的一个可写实例，这就是容器。

### 为什么 LLM 应用要使用 Docker？

LLM 应用通常依赖于一套特定的 Python 库（例如 LangChain、LlamaIndex、OpenAI 客户端、向量 (vector)数据库客户端、以及 FastAPI 等 Web 框架）和可能存在的系统依赖项。Docker 在管理这些需求方面表现出色：

1. **环境一致性：** Docker 确保您的应用在完全相同的环境中运行，无论容器部署在哪里。这消除了开发、测试和生产环境之间的不一致。
2. **依赖项管理：** 它将所有必需的库（在您的 `requirements.txt` 或类似文件中指定）打包在镜像中。您无需在目标机器上手动安装依赖项；Docker 在镜像构建过程中会处理好这一点。
3. **隔离性：** 容器彼此独立运行，也与宿主系统隔离，防止不同应用或库版本之间发生冲突。
4. **可移植性：** 在您的笔记本电脑上构建的 Docker 镜像，可以在任何安装了 Docker 的服务器或云平台运行，大幅简化了部署。
5. **可伸缩性：** 容器编排平台（如 Kubernetes）可以更轻松地根据需求扩展或缩减容器化应用。

### 为您的 LLM 应用创建 Dockerfile

`Dockerfile` 是一个文本文件，包含 Docker 用于构建镜像的一系列命令。我们来概述一下为 Python LLM 应用（例如使用 FastAPI 提供 API 端点的应用）创建 Dockerfile 的常见步骤：

1. **选择基础镜像：** 从官方 Python 基础镜像开始。选择与您开发时使用的 Python 版本一致的版本。为了可重现性，建议使用特定版本标签（例如 `python:3.10-slim`），而非 `latest`。`slim` 变体镜像更小，这通常是理想的。
2. **设置工作目录：** 定义容器内您的应用代码将存放的目录。
3. **复制依赖文件：** 将您的依赖文件（`requirements.txt`）复制到容器中。
4. **安装依赖项：** 运行 `pip install` 来安装您的依赖文件中列出的库。将依赖项安装与复制其余代码分开，可以让 Docker 缓存这一层。如果您的依赖项没有改变，Docker 在后续构建时就不需要重新安装它们，从而加快构建过程。
5. **复制应用代码：** 将您的应用源代码的其余部分复制到工作目录。
6. **暴露端口：** 如果您的应用运行 Web 服务器（如 FastAPI），请指定它监听的端口。
7. **定义运行时命令：** 指定容器启动时要执行的命令。这通常是运行您的 Python 应用的命令（例如 `uvicorn main:app --host 0.0.0.0 --port 8000`）。

这是一个 `Dockerfile` 示例：

```dockerfile
# Dockerfile

# 1. 使用官方 Python 运行时作为父镜像
FROM python:3.10-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 将 requirements 文件复制到容器的 /app 目录
COPY requirements.txt .

# 4. 安装 requirements.txt 中指定的任何所需包
# 使用 --no-cache-dir 以减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 5. 将应用代码的其余部分复制到容器的 /app 目录
COPY . .

# 6. 使端口 8000 在此容器外部可用
EXPOSE 8000

# 7. 定义 API 密钥的环境变量（将在 'docker run' 期间设置）
# 声明预期的环境变量是好习惯
ENV OPENAI_API_KEY="" 
# 添加其他必要的密钥（例如 ANTHROPIC_API_KEY）

# 8. 容器启动时运行 app.py
# 使用 0.0.0.0 使其可从容器外部访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**关于 API 密钥的重要说明：** 切勿将 API 密钥等敏感信息直接写入 `Dockerfile` 或源代码中。应使用环境变量，如 `ENV OPENAI_API_KEY=""` 所示。实际的密钥值应在运行容器时注入。

### 构建和运行 Docker 镜像

一旦您有了 `Dockerfile` 和您的应用代码（包括 `main.py` 和 `requirements.txt`），就可以构建镜像并运行容器了。

1. **构建镜像：** 在包含 `Dockerfile` 的目录中打开终端并运行：

   ```bash
   # -t 为镜像指定名称（例如 llm-app）和可选版本（例如 :latest）
   # . 指定构建上下文（当前目录）
   docker build -t llm-app:latest .
   ```

   Docker 将逐步执行您的 `Dockerfile` 中的指令。
2. **运行容器：** 镜像成功构建后，运行它：

   ```bash
   # -d 在分离模式下（后台）运行容器
   # -p 将宿主机的 80 端口映射到容器的 8000 端口
   # -e 在容器内设置环境变量 OPENAI_API_KEY
   # --name 为正在运行的容器指定一个可识别的名称
   docker run -d -p 80:8000 \
     -e OPENAI_API_KEY="your_actual_openai_api_key" \
     --name my-llm-app llm-app:latest
   ```

   将 `"your_actual_openai_api_key"` 替换为您的实际密钥。如果您的应用需要其他密钥，请添加更多 `-e` 标志。现在，您的 LLM 应用应该可以在 `http://localhost:80` 上可访问（如果远程部署，则使用服务器的 IP 地址）。

### 容器可视化

下图展示了 Docker 如何打包您的应用：

> 该图展示了 Docker 容器如何将 Python 代码、其依赖项和运行时打包。容器在宿主操作系统上独立运行，并与 LLM API 以及可能的向量 (vector)存储等外部服务通信。

### Docker 中 LLM 应用的考量

- **镜像大小：** LLM 依赖项有时会导致 Docker 镜像过大。使用 `.dockerignore` 文件可以从构建上下文 (context)中排除不必要的文件（如虚拟环境、测试数据、`.git` 目录）。对于更复杂的应用，可以考虑使用多阶段构建来保持最终镜像的精简。
- **资源管理：** LLM 推理 (inference)可能是 CPU 或内存密集型任务。在运行容器时，尤其是在生产环境中，应使用 Docker 选项（`--cpus`、`--memory`）或编排平台设置来配置资源限制（CPU、内存），以确保稳定性和公平的资源分配。
- **GPU 访问：** 如果您在本地运行需要 GPU 加速的模型（对于基于 API 的工作流较不常见，但有可能），您需要配置支持 GPU 的 Docker（例如使用 NVIDIA Container Toolkit），并在运行容器时指定 GPU 访问。

通过使用 Docker 将您的 Python LLM 应用容器化，您创建了一个标准化、可移植且隔离的运行环境。这大幅简化了部署过程，减少了与环境相关的错误，并为构建可伸缩、易维护的 LLM 驱动系统提供支撑。这是从开发脚本到操作可靠性的转变中一项重要的做法。

## 参考资料

- [Docker overview](https://docs.docker.com/get-started/overview/) — Docker Documentation (2024)
  提供Docker的基本概念，包括容器、镜像和Dockerfile，是理解容器化的良好起点。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  Publisher: Docker
  提供创建高效、可维护和安全Docker镜像的指南和技术，涵盖多阶段构建和缓存策略等主题。
- [Docker Deep Dive](https://leanpub.com/dockerdeepdive) — Nigel Poulton (2025)
  Publisher: Nigel Poulton Ltd.
  一本内容全面的书籍，对Docker基础知识、镜像创建、容器管理和实际部署场景进行了广泛介绍。
