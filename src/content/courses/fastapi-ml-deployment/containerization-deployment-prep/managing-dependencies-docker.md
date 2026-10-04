---
course: "fastapi-ml-deployment"
chapter: "containerization-deployment-prep"
lesson: "managing-dependencies-docker"
sourceId: 5350
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/managing-dependencies-docker"
title: "在 Docker 中管理 Python 依赖"
description: "处理 Dockerfile 中 Python 包安装的最佳实践。"
order: 5
plots: []
sourceHash: "dcb994fa4ba31b3089a50297b251e8eb40c5bc4f92cc357f5a3b0e3116f01d4f"
sourceCorrections: []
---

有效管理 Python 依赖是为 FastAPI 应用程序创建稳定且可重现的 Docker 镜像的基本要求。当你在本地运行 `pip install -r requirements.txt` 时，会将软件包安装到当前环境中。在 Docker 容器内部，每次构建镜像时都需一致地重复此过程。

最直接的方法是将 `requirements.txt` 文件复制到镜像中，然后运行 `pip install`：

```dockerfile
# Dockerfile - 基本依赖安装（效率较低）

# 基于一个 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制整个应用程序代码，包括 requirements.txt
COPY . .

# 安装依赖
# 问题：每当任何应用程序文件更改时，此步骤都会运行！
RUN pip install --no-cache-dir -r requirements.txt

# 运行应用程序的命令（示例）
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

虽然这种方法可行，但它与 Docker 的构建缓存有关联，存在一个明显的缺点。Docker 以分层方式构建镜像。`Dockerfile` 中的每条指令（如 `COPY`、`RUN`、`WORKDIR`）都会创建一个新层。如果自上次构建以来，某条指令及其输入文件没有改变，Docker 会重用上次构建的缓存层，从而大大加快过程。

在上述基本方法中，`COPY . .` 命令会复制你的*所有*项目文件。如果你的应用程序中*任何*文件发生更改（即使是 API 端点中微小的代码调整），`COPY . .` 命令的输入也会改变。这会使该层以及*所有后续层*的缓存失效，包括 `RUN pip install` 层。因此，即使 `requirements.txt` 本身没有改变，每次重建镜像时，Docker 都会重新安装所有 Python 依赖。对于可能包含 TensorFlow、PyTorch 或 scikit-learn 等大型库的机器学习 (machine learning)应用程序，这会大大增加你的开发和部署周期。

### 善用构建缓存以提高效率

为了优化这一点，你应该组织 `Dockerfile` 的结构，使其能充分利用层缓存。其方法是在复制应用程序其余代码*之前*复制并安装依赖。这样，依赖安装层只会在 `requirements.txt` 文件本身更改时才会被重建。

以下是改进后的结构：

```dockerfile
# Dockerfile - 优化后的依赖安装

# 基于一个 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 1. 首先只复制 requirements 文件
COPY requirements.txt .

# 2. 安装依赖
# 该层会被缓存，并且只在 requirements.txt 更改时才重新运行。
RUN pip install --no-cache-dir -r requirements.txt

# 3. 现在复制应用程序的其余代码
COPY . .

# 运行应用程序的命令（示例）
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

采用这种优化方法：

1. `COPY requirements.txt .`：这*只*复制了 requirements 文件。Docker 会缓存该层。
2. `RUN pip install --no-cache-dir -r requirements.txt`：这*仅*基于上一步中复制的 `requirements.txt` 文件安装依赖。只要 `requirements.txt` 在构建之间没有改变，Docker 就会重用包含所有已安装软件包的缓存层。`--no-cache-dir` 标志会告诉 `pip` 不要将其下载的软件包存储在缓存中，这有助于保持最终镜像尺寸更小，尽管如果需要重新下载软件包，安装本身可能会花费稍微长一点的时间。
3. `COPY . .`：这会复制应用程序的其余源代码（你的 Python 文件、模型工件等）。如果你只更改应用程序代码（例如，更新一个端点函数），则只有该层和任何后续层会被重建。可能耗时的依赖安装层会保持缓存状态。

这种简单的重新排序可以大大加快开发过程中的重建速度，尤其是在你频繁更改应用程序代码但其依赖未变时。

### 固定依赖版本

为了实现真正可重现的构建，请确保你的 `requirements.txt` 文件包含你依赖的固定版本。而不是像这样：

```
# requirements.txt (不太具体)
fastapi
uvicorn
pydantic
scikit-learn
joblib
```

请使用从你的工作开发环境中获取的特定版本，通常通过 `pip freeze` 获得：

```
# requirements.txt (固定版本)
fastapi==0.85.0
uvicorn[standard]==0.18.3
pydantic==1.10.2
scikit-learn==1.1.2
joblib==1.1.0
# ... 其他带特定版本的依赖
```

固定版本可以确保 `pip install -r requirements.txt` 始终在 Docker 容器内部安装与你在开发和测试期间使用的完全相同的软件包版本，从而避免因上游软件包更新而导致的意外行为。

### 虚拟环境与 Docker

一个常见的困惑点是在 Docker 容器内是否需要使用 Python 虚拟环境（如 `venv` 或 `conda`）。通常，这是不必要的，并且会增加额外步骤。Docker 容器提供自己的独立文件系统和进程空间。容器*本身*就是你的应用程序及其依赖的隔离环境。将软件包直接安装到*容器内*的系统 Python site-packages 目录（`pip install` 命令在以 root 用户或在基础环境中运行时默认就是这样做的）是标准做法，并能实现所需的隔离。

通过仔细管理何时以及如何在 `Dockerfile` 中安装依赖，善用 Docker 的层缓存，并固定依赖版本，你可以为 FastAPI 机器学习 (machine learning)应用程序创建更小、更稳定且构建速度更快的 Docker 镜像。这为跨不同环境的一致部署奠定了稳固的根基。

## 参考资料

- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  Publisher: Docker
  为Dockerfiles结构提供权威指导，包括如何利用构建缓存进行依赖管理。
- [Managing Application Dependencies](https://packaging.python.org/en/latest/guides/installing-packages/#managing-application-dependencies) — The PyPA (Python Packaging Authority) (2025)
  Publisher: Python Packaging Authority (PyPA)
  关于使用`requirements.txt`定义和固定Python依赖以实现可重复构建的官方指南。
