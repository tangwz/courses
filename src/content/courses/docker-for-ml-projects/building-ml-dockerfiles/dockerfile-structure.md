---
course: "docker-for-ml-projects"
chapter: "building-ml-dockerfiles"
lesson: "dockerfile-structure"
sourceId: 5245
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/dockerfile-structure"
title: "Dockerfile 的结构化方法"
description: "学习在机器学习 Dockerfile 中组织指令的有效方法，以提高清晰度和构建效率。"
order: 1
plots: []
sourceHash: "56eaf6b93a2bf05a7618c9cb18578bd41c6f7e2075b0ed85739b2dea3f991cdb"
sourceCorrections: []
---

`Dockerfile` 是构建容器镜像的蓝图。它是一个文本文件，包含一系列 Docker 遵循的指令，用于逐层构建镜像。虽然能正常运行的 `Dockerfile` 可以在很少规划的情况下编写，但一个*结构良好的* `Dockerfile` 会明显更容易阅读、维护、调试和优化，尤其对于复杂的机器学习 (machine learning)环境而言。可以将其比作编写清晰、有条理的代码；同样的原则也适用在这里。

`Dockerfile` 中指令的顺序非常重要，主要原因在于 Docker 的构建缓存。Docker 逐层构建镜像，`Dockerfile` 中的每条指令通常都会创建一个新层。当您重新构建镜像时，Docker 会检查该指令及其所依赖的文件自上次构建以来是否已更改。如果没有，Docker 会从缓存中复用现有层，而不是再次执行该指令。这能显著加快构建时间。

为充分发挥层缓存的效用，请将最不常变化的指令放在 `Dockerfile` 的前面，之后是变化较频繁的指令。机器学习项目的常见结构大致如下所示：

1. **基础镜像 (`FROM`):** 首先指定镜像的基础。这很少变化。
2. **元数据 (`LABEL`, `ARG`):** 定义标签或构建时参数 (parameter)。这些偶尔会变。
3. **系统依赖项 (`RUN apt-get update && apt-get install -y ...`):** 安装操作系统软件包（如 `git`、`wget` 或 C++ 构建工具）。这些依赖项一旦确定通常不会经常变化。
4. **Python 环境设置 (`RUN pip install --no-cache-dir --upgrade pip`):** 准备 Python 环境。
5. **Python 依赖项 (`COPY requirements.txt .`, `RUN pip install -r requirements.txt` 或 Conda 等效项):** 安装核心 Python 库（TensorFlow、PyTorch、Scikit-learn 等）。这是一个频繁变化的地方，但在运行安装命令*之前*复制 `requirements` 文件，能让 Docker 在依赖没有变化的情况下缓存这个可能耗时的安装步骤。
6. **复制项目代码 (`COPY ./src /app/src`):** 复制您的机器学习脚本、笔记本文件、工具文件等。这经常随着每次代码修改而变化。
7. **工作目录 (`WORKDIR`):** 设置后续命令的目录上下文 (context)。
8. **暴露端口 (`EXPOSE`):** 声明容器将监听的网络端口（与推理 (inference)服务器相关）。
9. **默认命令 (`CMD` 或 `ENTRYPOINT`):** 指定容器启动时要运行的命令。

请看这个简化图表，它说明了重新排序指令如何影响缓存使用：

> 层缓存对比。在“优良顺序”示例中，仅修改源代码（最后复制）能让 Docker 复用依赖项的缓存层，前提是 `requirements.txt` 未发生变化。在“欠佳顺序”中，过早复制所有代码通常会导致后续步骤（如依赖项安装）的缓存失效，即使仅修改了 Python 脚本也是如此。

通过以下方式可以提高命令顺序和可维护性：

- **使用注释：** 添加注释（`#`）来解释不明显的步骤、特定版本的选择或一组命令的目的。

  ```dockerfile
  # 使用官方 Python 运行时作为父镜像
  FROM python:3.9-slim

  # 设置容器中的工作目录
  WORKDIR /app

  # 安装某些 Python 库所需的系统依赖项
  # 示例：像 OpenCV 或 LightGBM 这样的库经常需要 libgomp1
  RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
   && rm -rf /var/lib/apt/lists/*

  # 使用 requirements 文件安装 Python 依赖项
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # 将本地代码复制到容器镜像中
  COPY ./src /app/src

  # 容器启动时运行的默认命令
  CMD ["python", "/app/src/train.py"]
  ```
- **分组命令：** 使用 `&&` 和行连接符（`\`）组合相关的 `RUN` 命令，以创建一个单个逻辑单元，并潜在地减少镜像层数。虽然现代构建器会优化层处理，但这种做法通过将相关操作放在一起来提高可读性。例如，更新软件包列表、安装软件包和清理 apt 缓存文件都可以放在一个 `RUN` 指令中。

  ```dockerfile
  # 可读性较差，且可能生成更多层
  RUN apt-get update
  RUN apt-get install -y package1
  RUN apt-get install -y package2
  RUN rm -rf /var/lib/apt/lists/*

  # 可读性更强，此逻辑操作对应单层
  RUN apt-get update && apt-get install -y --no-install-recommends \
      package1 \
      package2 \
   && rm -rf /var/lib/apt/lists/*
  ```
- **明确具体：** 避免在 `Dockerfile` 早期使用过于宽泛的 `COPY` 命令，如 `COPY . .`，尤其是在依赖安装之前。相反，明确地只复制特定步骤所需的内容（例如，在安装依赖项之前 `COPY requirements.txt .`，然后稍后 `COPY ./src /app/src`）。这种精细的方法能最大限度地利用缓存。

为 `Dockerfile` 采用一致的结构能让您的机器学习环境更可预测，构建过程更高效。它也使团队成员能更快理解镜像的构成，从而简化协作。在本章后续内容中，您将看到在选择基础镜像和管理特定机器学习项目依赖项时，此结构如何应用。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  Publisher: Docker
  阐述 Dockerfile 中的每条指令、其语法及使用方法。
- [Best practices for writing Dockerfiles](https://docs.docker.com/build/building/best-practices/) — Docker (2024)
  提供 Dockerfile 性能、大小和可维护性优化指南，包括构建缓存策略。
- [Docker for Developers: Build and Run Your Applications with Docker Containers](https://www.packtpub.com/product/docker-for-developers/9781801815777) — Peter Roemmler (2022)
  Publisher: Packt Publishing
  提供创建高效且易于维护的 Dockerfile 用于应用开发的实用建议和示例。
