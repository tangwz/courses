---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-inference"
lesson: "optimizing-image-size-multistage"
sourceId: 5317
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/optimizing-image-size-multistage"
title: "优化镜像大小：多阶段构建"
description: "在 Dockerfile 中使用多阶段构建来制作更小的、生产可用的推理镜像。"
order: 3
plots: []
sourceHash: "eb3426836c90373e07b694b340692ab1a66d2f5355b7a6d24e93c050b85c3d0b"
sourceCorrections: []
---

在部署机器学习 (machine learning)模型进行推理 (inference)时，Docker 镜像的大小是一个重要因素。较大的镜像下载时间更长，占用存储空间更多，并且可能通过包含不必要的构建工具或库来增加受攻击面。导致镜像过大的一个常见原因是包含了构建时依赖，例如编译器、头文件，甚至整个 SDK，这些是安装库所需的，但在运行最终应用时却不需要。

多阶段构建为此问题提供了一个有效办法。它们允许您在一个 Dockerfile 中使用多个 `FROM` 指令。每个 `FROM` 指令都可以使用不同的基础镜像，并开始一个新的构建“阶段”。您可以选择性地将产物（例如编译后的代码、已安装的依赖或模型文件）从一个阶段复制到另一个阶段，而将最终镜像中不需要的一切留下。

### 多阶段构建的工作原理

这种做法包括为构建和运行应用设定不同的阶段：

1. **构建阶段：** 此阶段通常以一个较大的基础镜像开始，其中包含所有需要的编译器、构建工具和开发头文件。您在此阶段执行安装依赖（可能涉及编译）、下载模型或预处理数据等任务。您通常使用 `AS <stage_name>`（例如 `AS builder`）来给这个阶段命名。
2. **最终阶段：** 此阶段以一个适合运行时的最小基础镜像开始（例如 `python:3.9-slim` 或一个 distroless 镜像）。它只包含运行您的推理 (inference)服务所必需的组件。
3. **复制产物：** 重要一步是在最终阶段使用 `COPY --from=<stage_name>` 指令。此命令将指定的前一阶段（例如 `builder`）中的特定文件或目录复制到最终阶段的文件系统中。

这个过程确保最终镜像只包含运行时应用、其直接依赖项以及任何需要的数据文件，同时舍弃中间构建环境及其附带的冗余。

### 示例：优化 Python 推理 (inference)服务

考虑一个使用 FastAPI 构建的推理服务，它需要可能带有复杂构建依赖的库。一个多阶段 Dockerfile 可能像这样：

```dockerfile
# 阶段 1: 构建器阶段，包含构建工具
FROM python:3.9 AS builder

WORKDIR /app

# 如有需要，安装构建必备工具（Debian系镜像示例）
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# 先只复制 requirements 文件，以利用 Docker 缓存
COPY requirements.txt .

# 安装所有依赖，包括构建时依赖
# 在构建阶段使用虚拟环境有助于隔离软件包
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码的其余部分
COPY . .

# 可选：如果模型文件没有随代码复制，则在此处下载
# RUN python download_model.py

# 阶段 2: 最终阶段，使用精简运行时环境
FROM python:3.9-slim

WORKDIR /app

# 从构建器阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 复制应用代码（如有需要请调整路径）
COPY --from=builder /app /app

# 如果模型文件在构建器阶段已下载或属于应用代码的一部分，则在此处复制
# COPY --from=builder /app/models /app/models

# 使端口 80 在容器外部可用
EXPOSE 80

# 定义环境变量
ENV NAME

# 设置路径以包含虚拟环境的二进制文件
ENV PATH="/opt/venv/bin:$PATH"

# 运行应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

在此示例中：

- `builder` 阶段使用标准 `python:3.9` 镜像，将 `requirements.txt` 中的依赖安装到虚拟环境 (`/opt/venv`) 中，并复制应用代码。
- 最终阶段从小得多的 `python:3.9-slim` 镜像开始。
- `COPY --from=builder /opt/venv /opt/venv` 将包含所有已安装 Python 软件包的*整个*虚拟环境从构建器阶段复制到最终阶段。
- `COPY --from=builder /app /app` 复制应用源代码。
- 最终镜像包含精简的 Python 运行时、虚拟环境中需要的已安装软件包以及应用代码，但*不包含*构建工具或来自 `builder` 阶段的中间层。

> 此图示说明了在多阶段 Docker 构建中，如何将特定产物（虚拟环境、应用代码）从较大的构建阶段复制到较小的最终运行时阶段。

### 多阶段构建对推理 (inference)的好处

- **减小镜像大小：** 这是主要优点。移除构建工具、临时文件和不必要的库可以大幅缩小最终镜像的大小，从而加快部署并降低存储成本。
- **提高安全性：** 通过将构建工具和开发库从最终镜像中排除，您可以减少潜在的攻击面。组件越少意味着潜在漏洞越少。
- **清晰分离：** 它使得构建环境和运行时环境之间有了清晰的分离，让 Dockerfile 更易于理解和维护。
- **优化缓存：** Docker 可以缓存各个阶段。如果构建阶段的依赖没有改变，Docker 可以复用缓存的构建阶段，即使最终阶段需要修改，也能加快后续构建速度。

### 注意事项

- **识别产物：** 您需要仔细确定哪些文件和目录是运行时所需，并确保使用 `COPY --from` 正确复制它们。这可能包含已安装的软件包（例如 `site-packages` 目录或虚拟环境）、编译后的二进制文件、静态资源和模型文件。检查中间构建阶段容器的文件系统（`docker run --rm -it <builder_image_id> bash`）会很有帮助。
- **基础镜像选择：** 为每个阶段选择合适的基础镜像。构建阶段可能需要完整的操作系统发行版或特定的 SDK 镜像，而最终阶段应尽可能精简（例如 `slim`、`alpine` 或 `distroless`）。

多阶段构建是创建生产可用容器镜像的标准做法，对于效率和安全性是重要考量的机器学习 (machine learning)推理 (inference)服务来说尤其适用。通过将构建时需求与运行时必要项分离，您可以创建更精简、更快速、更安全的容器来服务您的模型。

## 参考资料

- [Use multi-stage builds](https://docs.docker.com/build/building/multi-stage/) — Docker Documentation (2024)
  本官方指南介绍并解释了 Docker 的多阶段构建功能，详细说明了其指令和实现高效镜像创建的模式。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书涵盖了部署机器学习模型的实践方面，其中容器化和优化推理服务镜像大小是重要的操作考量。
