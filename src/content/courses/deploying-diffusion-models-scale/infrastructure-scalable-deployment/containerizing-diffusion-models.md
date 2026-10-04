---
course: "deploying-diffusion-models-scale"
chapter: "infrastructure-scalable-deployment"
lesson: "containerizing-diffusion-models"
sourceId: 5404
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment/containerizing-diffusion-models"
title: "扩散模型Docker容器化"
description: "将扩散模型及其依赖项打包到可移植的 Docker 容器中。"
order: 1
plots: []
sourceHash: "465827cf1243d939932dbdc72c630445b7146bd8d99ab39d88dcbe55cd83dfc6"
sourceCorrections: []
---

部署像扩散模型这样复杂的应用程序需要一致且可复现的环境。容器化，特别是使用 Docker，为此类一致性提供了根本支持，贯穿开发、测试和生产阶段。它将应用程序代码、运行时、系统工具、库和配置文件打包成一个单一、可移植的单元，称为容器镜像。

### 为何扩散模型使用 Docker？

扩散模型通常带有很多依赖项：特定版本的深度学习 (deep learning)框架（PyTorch, TensorFlow）、与目标硬件兼容的 CUDA 库、Python 包（如 `diffusers`、`transformers`、`accelerate`），以及可能的系统级工具。在不同机器或环境中手动管理这些依赖项容易出错且耗时。Docker 通过以下方式解决此问题：

1. **封装依赖项：** Docker 镜像包含运行模型推理 (inference)服务所需的一切。这解决了“在我的机器上能运行”的问题。
2. **环境一致性：** 保证无论容器在何处运行，从开发者的笔记本电脑到云虚拟机或 Kubernetes Pod，运行时环境都相同。
3. **隔离：** 容器在隔离环境中运行，防止同一主机上运行的不同应用程序或模型版本之间发生冲突。
4. **可移植性：** Docker 容器可在任何支持 Docker 的系统上运行，简化了跨不同云提供商或本地基础设施的部署。
5. **编排的支撑：** 容器化是使用 Kubernetes 等容器编排平台的先决条件，这些平台对大规模部署管理很重要（本章后续会涉及）。

### 构建用于推理 (inference)的 Dockerfile

创建 Docker 镜像的蓝图是 `Dockerfile`。它包含一系列定义环境的指令。让我们查看一个扩散模型推理服务的典型结构，它通常围绕 FastAPI 或 Flask 等 Web 框架构建。

```dockerfile
# 1. 基础镜像 - 选择包含所需 CUDA/cuDNN 版本的镜像
# 使用官方 NVIDIA CUDA 镜像的示例
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.10
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu22.04 AS base

# 设置环境变量，避免安装时出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 2. 安装系统依赖项和 Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    git \
    # 添加任何其他必要的系统软件包（例如，build-essential, cmake）
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 将 python3 链接到 python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

# 升级 pip
RUN python -m pip install --upgrade pip

# 3. 设置应用程序目录
WORKDIR /app

# 4. 安装 Python 依赖项
# 首先复制 requirements，以利用 Docker 层缓存
COPY requirements.txt .
# 考虑优化安装；例如，如果依赖项已仔细处理，可以使用 --no-deps
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. 复制应用程序代码和模型占位符（如果未下载）
COPY ./src /app/src
# 可选地复制脚本、配置等
COPY ./scripts /app/scripts
COPY ./config /app/config

# 确保模型目录存在，如果稍后下载
RUN mkdir -p /app/models

# 6. 暴露端口（与推理服务器使用的端口匹配）
EXPOSE 8000

# 7. 定义 Entrypoint/Command
# 示例：使用 uvicorn 运行 FastAPI 服务器
# 假设您的主应用程序对象位于 /app/src/main.py 中，名为 'app'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile 的主要注意事项：**

- **基础镜像选择：** 这很重要。使用官方 NVIDIA CUDA 镜像（例如，`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`）可确保与主机上的 GPU 驱动程序兼容。如果您不需要最终容器内的完整开发工具包，请选择 `runtime` 镜像以减小大小。将 CUDA/cuDNN 版本与深度学习 (deep learning)框架所需的版本以及目标硬件支持的版本相匹配。
- **依赖项安装：** 使用 `apt-get install -y --no-install-recommends` 并在之后进行清理（`apt-get clean`, `rm -rf /var/lib/apt/lists/*`）以尽量减小镜像大小。通过先复制 `requirements.txt` 并安装依赖项，再复制应用程序代码，可利用 Docker 的层缓存功能。
- **Python 环境：** 明确定义 Python 版本并确保 `pip` 是最新的。在 Docker 内部使用虚拟环境通常没有必要，因为容器本身就提供了隔离。
- **工作目录：** 设置 `WORKDIR` 以提高清晰度和一致性。
- **暴露端口：** `EXPOSE` 记录应用程序监听的端口，但您在运行容器时仍需使用 `docker run -p <host_port>:<container_port>` 进行映射。
- **CMD 与 ENTRYPOINT：** `CMD` 为运行中的容器提供默认参数 (parameter)，这些参数可以轻松覆盖。`ENTRYPOINT` 配置一个将作为可执行文件运行的容器；`CMD` 可以为 `ENTRYPOINT` 提供默认参数。对于运行 Web 服务器，`CMD` 通常已足够。

### 在容器中处理大型模型权重 (weight)

扩散模型的权重从数百兆字节到数千兆字节不等。您如何将这些权重包含在容器化环境中，对镜像大小、构建时间和启动延迟有重要影响。

1. **烘焙到镜像中：** 使用 Dockerfile 中的 `COPY` 指令将模型文件直接复制到镜像中。
   - *优点：* 镜像自包含，镜像拉取后启动最快。
   - *缺点：* 镜像体积非常大，镜像构建和传输（推/拉）缓慢，更新模型需要重建镜像。
2. **通过卷挂载：** 将模型权重存储在主机或网络文件系统（如 AWS EFS, GCP Filestore）上，并在运行时使用 `docker run -v /path/on/host:/path/in/container ...` 将此位置挂载到容器中。
   - *优点：* 保持镜像体积小，无需重建即可轻松更新模型。
   - *缺点：* 对主机/外部存储产生依赖，路径管理可能复杂，如果依赖主机路径则可移植性较差。
3. **运行时下载：** 在容器的入口点脚本或应用程序启动代码中添加逻辑，如果模型权重在容器内不存在，则从中心存储库（例如 Hugging Face Hub, AWS S3, GCS Bucket）下载模型权重。
   - *优点：* 镜像体积最小，模型更新灵活，将模型工件与应用程序镜像分离。
   - *缺点：* 增加容器启动时间（“冷启动”），启动时需要网络访问，需要处理下载失败的错误。

> 将模型权重包含在 Docker 容器中的策略。烘焙会导致镜像较大，挂载依赖外部存储，而下载会增加启动延迟。

最佳方法通常取决于具体使用场景。对于开发，挂载可能最简单。对于生产，运行时下载结合缓解冷启动的策略（如实例预热或保持最小数量的实例运行）是一种常见模式，尤其是在使用 Kubernetes 等编排系统时。

### 构建和运行镜像

`Dockerfile` 准备就绪后，您可以使用以下命令构建镜像：

```bash
# 构建镜像，并打标签以便引用
docker build -t my-diffusion-service:latest .

# 或者如果 Dockerfile 不在默认位置，则指定其路径
docker build -f path/to/your/Dockerfile -t my-diffusion-service:v1.0 .
```

要在本地运行容器并测试服务（假设它监听端口 8000）：

```bash
# 在分离模式 (-d) 下运行，将主机端口 8080 映射到容器端口 8000 (-p)
# 并且可选地传递环境变量 (-e) 或挂载卷 (-v)
docker run -d -p 8080:8000 \
  --gpus all \ # 请求访问主机 GPU（需要 NVIDIA Container Toolkit）
  -e MODEL_ID="stabilityai/stable-diffusion-2-1-base" \
  --name diffusion-app \
  my-diffusion-service:latest
```

请注意 `--gpus all` 标志。这需要主机上安装 NVIDIA Container Toolkit，允许容器访问主机的 GPU。在容器内管理 GPU 资源将在下一节讨论。

运行后，您可以测试端点，例如使用 `curl`：

```bash
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A photograph of an astronaut riding a horse"}'
```

### 优化镜像大小和构建时间

大型 Docker 镜像构建、推送、拉取和扫描都慢。特别是对于包含许多依赖项的机器学习 (machine learning)模型，优化很重要。

- **多阶段构建：** 在 Dockerfile 中使用多个 `FROM` 语句。一个阶段可用于构建依赖项或编译代码，后续阶段仅将必要的工件复制到更小的最终镜像中（通常基于 `runtime` 基础镜像而非 `devel`）。
- **最小化层数：** 每个指令（`RUN`、`COPY`、`ADD`）都会创建一个层。使用 `&&` 和反斜杠（`\`）组合相关的 `RUN` 命令，以减少层数。
- **.dockerignore 文件：** 在构建上下文 (context)目录（通常是 Dockerfile 所在的位置）中创建一个 `.dockerignore` 文件，以排除镜像中不需要的文件和目录（例如，`.git`、`__pycache__`、本地数据集、虚拟环境文件夹）。
- **基础镜像选择：** 较小的基础镜像（如 `python:3.10-slim-bullseye` 或基于 Alpine 的镜像）可以显著减小大小，但请彻底测试，因为它们缺少常用库，可能导致与复杂软件包的兼容性问题。对于 GPU 使用，NVIDIA 的运行时镜像通常是最佳起点，尽管它们体积较大。

### 安全最佳实践

- **非 Root 用户：** 避免以 `root` 用户运行容器。在 Dockerfile 中创建一个专用用户和组，并在最终的 `CMD` 或 `ENTRYPOINT` 之前使用 `USER` 指令切换到该用户。

```dockerfile
# ... 前面的指令 ...

# 创建非 root 用户和组
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --shell /bin/bash --create-home appuser

# 设置应用程序目录的所有权
RUN chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 定义 Entrypoint/Command（以 appuser 身份运行）
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **漏洞扫描：** 将镜像扫描工具（例如 Trivy, Docker Scout, 云提供商扫描器）集成到您的 CI/CD 流水线中，以检测操作系统软件包和语言库中的已知漏洞。
- **秘密管理：** 不要在 Dockerfile 或镜像层中硬编码秘密（API 密钥、数据库密码）。使用在运行时注入的环境变量或编排器或云平台提供的专用秘密管理系统。

将您的扩散模型推理 (inference)服务用 Docker 容器化是实现可扩展和可靠部署的根本一步。它将您的应用程序及其复杂的依赖项打包成一个可移植的单元，准备好由 Kubernetes 等编排系统管理，接下来我们将讨论这些容器化环境中的 GPU 资源管理的具体考虑事项。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker Inc. (2024)
  涵盖Docker基本概念、Dockerfile语法、镜像构建和容器运行，与本节核心内容直接相关。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Inc. (2024)
  Publisher: Docker Inc.
  提供优化Docker镜像大小、构建时间以及增强安全性的详细指南，与镜像构建直接相关。
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) — NVIDIA (2024)
  提供在Docker容器中启用GPU加速的信息，对深度学习工作负载至关重要。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供设计稳健ML系统的方法，包括模型部署和容器化策略的实践方面。
