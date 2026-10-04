---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-inference"
lesson: "reducing-inference-dependencies"
sourceId: 5320
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/reducing-inference-dependencies"
title: "减少推理时的依赖项"
description: "从最终推理镜像中移除不必要的构建时依赖项的方法。"
order: 4
plots: []
sourceHash: "0f5afb00c57f470869e7da93c8b4ea5aed5723a67af1de85e892bf13635b5f51"
sourceCorrections: []
---

部署机器学习 (machine learning)模型进行推理 (inference)时，容器镜像的效率和安全性是很重要的考量。大型镜像会占用更多磁盘空间，下载和启动时间更长，并且可能包含不必要的软件，从而增大受攻击面。因此，减少最终推理镜像中的依赖项是一项重要的优化措施。这通常需要借鉴多阶段构建所介绍的方法。

虽然开发或训练环境可能需要各种工具、编译器、数据处理库（如 pandas）、绘图库、完整的机器学习框架安装以及测试工具，但推理容器通常所需更少。它的主要工作通常是加载预训练 (pre-training)模型并通过轻量级 Web API 运行预测。

### 识别不必要的依赖项

第一步是仔细分析您的推理 (inference)服务 *实际* 运行所需的内容。通常可以从最终推理阶段排除的常见依赖项包括：

- **构建工具：** 编译器（`gcc`、`g++`）、构建系统（`make`、`cmake`）以及包管理器缓存（`apt`、`pip`、`conda`）。这些是 *构建* 软件或安装包所必需的，但对于 *运行* 编译后的应用或已安装的库则非必需。
- **开发头文件和库：** 以 `-dev` 或 `-devel` 结尾的包（例如 `python3-dev`、`libssl-dev`）是编译某些带有 C 扩展的 Python 包所必需的，但在运行时很少需要。
- **完整数据科学库：** 虽然您在训练和实验期间可能大量使用 Pandas 或 Matplotlib 等库，但您的推理代码可能只需要 NumPy 进行数值运算或 ML 框架中的特定函数。如果您只需要大型库的一部分，请考虑是否可以重构您的推理代码以避免这种大依赖项。
- **测试框架：** 像 `pytest` 或 `unittest` 这样的库对于开发是必需的，但在生产推理镜像中没有位置。
- **未使用的代码和资源：** 确保只有特定的推理脚本、模型产物和必要的配置文件被复制到最终镜像中，排除数据集、Jupyter Notebook 或训练脚本。

### 最小化依赖项的策略

几种通常结合使用的技术有助于创建精简的推理 (inference)镜像：

#### 1. 积极利用多阶段构建

多阶段构建是将构建时需求与运行时要求分离的主要机制。

- **构建阶段：** 使用包含所有必要构建工具的基础镜像（例如 `python:3.9`，它是基于 Debian 并包含构建必需品的）。使用 `pip` 或 `conda` 安装依赖项，编译任何必要的代码，并准备您的应用。
- **最终阶段：** 从最小的基础镜像开始（例如 `python:3.9-slim-buster`，甚至 `python:3.9-alpine`，同时要留意 Alpine 的 `musl libc` 可能存在的兼容性问题）。 *只* 复制构建阶段所需的产物：
  - 已安装的 Python 包（例如虚拟环境的 `site-packages` 目录）。
  - 您编译的应用代码或推理脚本。
  - 已训练的模型文件。
  - 任何必要的配置文件。

```dockerfile
# ---- 构建阶段 ----
FROM python:3.9 AS builder

WORKDIR /app

# 如果需要，安装构建依赖项（例如，用于 C 扩展）
# RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制 requirements 并安装包
COPY requirements.txt .
# 使用 --no-cache-dir 减小层大小
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./src ./src
COPY ./models ./models

# ---- 最终阶段 ----
FROM python:3.9-slim-buster

WORKDIR /app

# 只从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 只复制必要的应用代码和模型
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models

# 设置路径以使用虚拟环境
ENV PATH="/opt/venv/bin:$PATH"

# 如果是 API，暴露端口
EXPOSE 8000

# 定义运行应用的命令
CMD ["python", "src/inference_api.py"]
```

在此示例中，像 `gcc`（如果安装在构建器中）和 `pip` 缓存之类的工具都留在 `builder` 阶段，不会出现在最终更精简的镜像中。

#### 2. 选择最小基础镜像

您的 *最终* 阶段的基础镜像对大小有很大影响。请考虑以下选项：

- **`python:<version>-slim`：** 这些基于 Debian，但与标准 `python:<version>` 镜像相比，删除了许多非必需的操作系统包。它们在大小和兼容性之间提供了良好的平衡。
- **`python:<version>-alpine`：** 基于轻量级的 Alpine Linux 发行版。这些镜像小得多，但使用 `musl libc` 而非更常见的 `glibc`。这有时会导致细微的问题或与预期 `glibc` 的预编译 Python wheel 存在不兼容。如果选择 Alpine，请彻底测试。
- **Distroless 镜像：** Google 的 Distroless 镜像只包含您的应用及其运行时依赖项。它们不包括包管理器、shell 或其他标准操作系统工具，极大地减小了受攻击面和大小。构建这些镜像通常需要更复杂的多阶段设置，细致地复制应用二进制文件和依赖项。

#### 3. 精简包安装

在推理环境的 `requirements.txt` 中要明确且尽量精简。

- 固定版本（`==`）以确保可重复性，并避免意外引入更新、可能更大的依赖项版本。
- 只包含推理严格必需的包。审查您的推理代码，并移除预测路径中未使用的导入/依赖项。
- 如果可用，考虑使用框架特定的推理运行时（例如，如果您使用 TFLite 模型，则使用 `tensorflow-lite-runtime` 而非完整的 `tensorflow` 包）。
- 使用 `pip install --no-cache-dir` 来避免在镜像层内缓存已下载的包，从而减小其大小。
- 如果没有有效使用多阶段构建，请清理包管理器缓存（例如，对于基于 Debian/Ubuntu 的镜像，执行 `apt-get clean && rm -rf /var/lib/apt/lists/*`）。多阶段构建通常会使这对最终镜像的必要性降低。

#### 4. 分析您的镜像层

`docker history <image_name>` 等工具可以显示镜像中每个层的大小贡献。像 `docker-explore` 这样的第三方工具提供了一种更具交互性的方式来检查镜像层并发现对大小贡献最大的文件。这有助于找到可能无意中复制的大文件或目录，或者未清理的缓存。

通过认真运用这些技术，特别是多阶段构建和仔细的依赖项选择，您可以大幅减小推理容器镜像的大小和复杂性。这会带来更快的部署、更高的安全性和生产环境中更高效的资源利用。

## 参考资料

- [Use multi-stage builds](https://docs.docker.com/build/building/multi-stage/) — Docker (2024)
  Publisher: Docker
  这份官方文档详细指导了如何实现多阶段 Docker 构建，这是一种通过分离构建时依赖项和运行时需求来创建更精简、更安全的容器镜像的基本技术。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker (2024)
  Publisher: Docker
  Docker 提供的一份全面指南，介绍如何优化 Dockerfile 创建，以生成更小、更高效、更安全的镜像，涵盖了选择基础镜像、最小化层和清理不必要文件等方面。
- [Distroless containers](https://github.com/GoogleContainerTools/distroless) — GoogleContainerTools (2024)
  Publisher: GoogleContainerTools
  此项目仓库解释了 Distroless 镜像的概念和实现，它通过仅包含应用程序及其运行时依赖项来提供极其精简的基础镜像，显著减小了镜像大小和攻击面。
- [Designing Machine Learning Systems: New Rules for an AI-Driven World](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于 MLOps 和机器学习模型生产化工程挑战的著作，为高效模型部署和容器化提供了实用建议和架构考量。
