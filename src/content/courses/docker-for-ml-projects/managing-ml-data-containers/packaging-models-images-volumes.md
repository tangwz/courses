---
course: "docker-for-ml-projects"
chapter: "managing-ml-data-containers"
lesson: "packaging-models-images-volumes"
sourceId: 5284
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/packaging-models-images-volumes"
title: "将模型打包到镜像中与卷的使用"
description: "讨论将模型内置到镜像中与通过卷加载模型之间的权衡。"
order: 6
plots: []
sourceHash: "226d334155b0dfd612e644f969c7da8145ca3cd89aaf3f9c15c733ee951c39f5"
sourceCorrections: []
---

一旦你训练好了一个机器学习 (machine learning)模型，就需要一种方法让容器化应用能够访问它，无论是为了进一步的训练、评估还是推理 (inference)。如本章前面所述，容器的文件系统通常是临时的。简单地将模型文件保存在运行中的容器内，通常意味着当它停止时文件就会丢失。

持久化模型并在运行时使其可用，需要从两种主要策略中选择，每种策略都对你的工作流程、镜像大小和部署过程有不同的影响：

1. **直接将模型打包到 Docker 镜像中。**
2. **使用 Docker 卷或绑定挂载从外部源加载模型。**

我们来分析每种方法的特点、优点和缺点。

### 将模型打包到 Docker 镜像中

使用 Dockerfile 中的 `COPY` 或 `ADD` 等指令，可以在构建过程中将模型文件直接整合到 Docker 镜像中。

```dockerfile
# Dockerfile 示例片段
FROM python:3.9-slim

WORKDIR /app

# 复制应用程序代码
COPY ./app /app

# 将训练好的模型文件复制到镜像中
COPY ./models/sentiment_model.pkl /app/models/sentiment_model.pkl

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 运行应用程序的命令（应用程序从 /app/models/ 加载模型）
CMD ["python", "serve.py"]
```

在这种情况下，`sentiment_model.pkl` 文件成为镜像层的一部分。

**优点：**

- **独立部署：** Docker 镜像成为一个单一、不可变的产物，包含运行应用所需的一切：代码、依赖项和特定的模型版本。这简化了分发和部署，因为你只需管理镜像即可。
- **版本一致性：** 模型版本与镜像版本固有绑定。回滚到先前的镜像版本会自动回滚模型，确保代码和模型之间的一致性。
- **容器启动更快：** 容器启动时，模型文件已存在于容器的文件系统中。应用程序可以立即加载模型，无需等待外部挂载或网络下载。

**缺点：**

- **镜像大小增加：** 机器学习 (machine learning)模型，特别是深度学习 (deep learning)模型，可能非常大（数百兆甚至数千兆字节）。直接包含它们会显著增大镜像大小。这会导致镜像构建、推送和拉取变慢，注册表中的存储成本增加，并且可能在没有缓存镜像的节点上导致容器启动变慢。
- **模型更新需要重新构建镜像：** 如果你重新训练并生成了新版本的模型，即使应用程序代码没有改变，也必须重新构建 Docker 镜像来包含它。这个重建过程可能耗时，并可能导致仅依赖于代码的组件进行不必要的重新部署。
- **灵活性降低：** 快速切换模型（例如，用于 A/B 测试或为不同租户使用不同模型）变得麻烦，因为每个变体都需要单独的镜像构建或复杂的入口点逻辑。

**何时使用：** 这种方法通常适用于：

- 模型较小，对镜像大小的影响可接受的情况。
- 推理 (inference)服务，其中模型和应用程序代码联系紧密并同时更新。
- 优先考虑部署简单性和不变性的情况，其中为模型更新而重建的开销是可控的。

### 通过卷或绑定挂载加载模型

这种替代策略将模型文件与 Docker 镜像分离。镜像仅包含应用程序代码及其依赖项。在运行时，模型文件通过 Docker 卷（用于生产/持久存储）或绑定挂载（通常用于开发）在容器内部可用。

应用程序代码被设计为从容器内的特定路径加载模型，该路径对应于卷或绑定挂载的挂载点。

```dockerfile
# Dockerfile 示例片段（模型未复制）
FROM python:3.9-slim

WORKDIR /app

# 复制应用程序代码
COPY ./app /app

# 模型稍后将挂载到 /app/models
# 如果应用需要，确保目录存在
RUN mkdir -p /app/models

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 运行应用程序的命令
# 假设模型将在 /app/models/sentiment_model.pkl 可用
CMD ["python", "serve.py"]
```

要运行此命令，你需要使用带有 `-v` 标志的 `docker run`：

```bash
# 使用名为 'ml_models' 的 Docker 卷
docker run -d -p 8000:8000 \
  -v ml_models:/app/models \
  my_ml_app:latest

# 使用绑定挂载（挂载本地 ./models 目录）
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  my_ml_app:latest
```

**优点：**

- **更小、更精简的镜像：** 镜像仅包含代码和依赖项，从而实现更快的构建、推送、拉取和更低的存储成本。
- **模型更新解耦：** 模型可以独立于应用程序代码进行更新。你可以将新的模型文件放入卷或主机目录中，随后的容器运行（甚至正在运行的容器，取决于应用程序逻辑）都可以获取新模型，而无需重新构建镜像或重新部署。
- **灵活性：** 通过更改挂载到容器中的数据源，可以轻松切换模型。这对于实验、A/B 测试或动态服务不同模型很有益。在多个容器之间共享模型也直截了当。

**缺点：**

- **需要外部管理：** 你需要一个单独的流程来管理模型文件的生命周期：存储它们、版本控制它们，并确保正确的版本放置到容器使用的卷或主机路径中。这增加了操作的复杂性。
- **运行时依赖：** 容器的成功运行取决于卷或绑定挂载在运行时是否正确配置和填充。挂载错误或模型文件丢失可能导致应用程序失败。
- **潜在的启动延迟：** 如果模型需要在容器启动前复制到卷中，或者卷驱动程序涉及网络访问，与将模型内置到镜像中相比，可能会有轻微的延迟。

**何时使用：** 这种方法通常更适合于：

- 模型很大，将其包含在镜像中不切实际的情况。
- 模型更新频繁且独立于应用程序代码的场景。
- 开发环境，其中绑定挂载允许模型更改即时反映。
- 需要模型共享或动态模型加载的情况。
- 工作流程中，模型产物在外部存储（如云存储桶 S3、GCS、Azure Blob Storage）中管理并获取到卷中。

### 选择正确的策略

将模型打包到镜像内部与通过卷加载模型之间的决策需要权衡。考虑以下因素：

- **模型大小：** 很大的模型强烈推荐使用卷。
- **模型更新频率：** 频繁且独立的模型更新推荐使用卷。
- **代码/模型关联性：** 如果代码版本始终与特定模型版本绑定，嵌入 (embedding)可以简化版本管理。如果它们独立演进，卷提供更大的灵活性。
- **操作复杂性：** 嵌入简化了部署产物（仅是镜像），但使更新复杂化。卷简化了更新，但需要外部模型管理。
- **团队工作流程：** 模型如何训练、版本控制和批准部署？将策略与你的 MLOps 实践对齐 (alignment)。
- **环境：** 绑定挂载非常适合本地开发迭代。卷通常用于预生产或生产环境的持久化。嵌入式模型可能用于高度重视不变性的生产推理 (inference)服务。

> 模型打包策略比较：将模型嵌入镜像中与通过卷挂载模型。

没有唯一的“最佳”答案；最佳选择很大程度上取决于你的具体应用、模型特性和操作环境。通常，团队为了简单起见会从嵌入模型开始，然后随着模型变大或更新频率与应用程序代码不同步时，再转为使用卷。理解这些权衡可以帮助你为容器化的机器学习 (machine learning)模型选择最有效的数据管理策略。

## 参考资料

- [Use volumes](https://docs.docker.com/storage/volumes/) — Docker Documentation (2024)
  Publisher: Docker
  Docker 卷的官方指南，解释了它们的用途、用法以及如何为容器提供持久存储，这对于外部模型加载至关重要。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  提供优化 Docker 镜像的指导原则，包括最小化镜像大小和构建时间，这与嵌入模型导致镜像大小增加的缺点相关。
- [Introducing MLOps: How to go from Model to Production](https://www.oreilly.com/library/view/introducing-mlops/9781098104860/) — Mark Treveil, Nicolas Omont, Aurélien Géron, Clemens Mewald, Michel Tessier, Alexey Grigorev (2022)
  Publisher: O'Reilly Media
  一本关于 MLOps 实践的综合书籍，其中包括模型服务、部署策略以及在生产环境中管理模型工件的讨论。
- [MLOps: A guide to operations for machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  Google Cloud 的官方 MLOps 指南，涵盖部署和管理 ML 系统的原则和实践，包括使用容器化包装和提供模型的注意事项。
