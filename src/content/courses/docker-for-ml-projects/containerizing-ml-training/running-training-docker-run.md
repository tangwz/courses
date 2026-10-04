---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-training"
lesson: "running-training-docker-run"
sourceId: 5297
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/running-training-docker-run"
title: "使用 `docker run` 运行训练任务"
description: "使用 `docker run` 命令执行容器化训练脚本。"
order: 3
plots: []
sourceHash: "17abfacf5e566154a3aaf39d7216f9a92acb8f3fa6e556dab1ca425ba217f725"
sourceCorrections: []
---

要执行机器学习 (machine learning)训练过程，`docker run` 是从 Docker 镜像创建并运行容器实例的`主要`命令。此命令可以在隔离环境中执行您的训练脚本及其所有依赖项。使用 `docker run` 可确保训练环境与 Dockerfile 中指定的配置精确匹配，从而避免主机上的差异，并提高结果的可重现性。

本节主要介绍如何有效地使用 `docker run` 命令来启动、管理和配置容器化的机器学习训练任务。

### 基本训练执行

其核心功能是，`docker run` 会从指定镜像创建一个新的容器并启动它。如果您的 Dockerfile 包含指向您训练脚本的 `ENTRYPOINT` 或 `CMD` 指令，那么运行容器可能非常简单，只需：

```bash
docker run your-ml-training-image:latest
```

此命令指示 Docker 进行以下操作：

1. 查找名为 `your-ml-training-image` 且标签为 `latest` 的镜像。
2. 根据此镜像创建一个新容器。
3. 启动容器，执行 Dockerfile 中 `ENTRYPOINT` 或 `CMD` 指定的命令。

通常，您会希望为训练脚本提供特定参数 (parameter)，例如超参数 (hyperparameter)或数据路径，甚至在镜像中运行不同的脚本。您可以通过在镜像名称后添加命令及其参数来实现这一点：

```bash
# 运行 train.py 并带上特定参数
docker run your-ml-training-image:latest python train.py --epochs 20 --batch-size 64

# 运行另一个脚本，例如数据预处理脚本
docker run your-ml-training-image:latest python preprocess_data.py --input /raw_data --output /processed_data
```

当您在镜像名称后提供命令时，它通常会覆盖 Dockerfile 中的 `CMD` 指令。如果定义了 `ENTRYPOINT`，您提供的命令可能会作为参数传递给 `ENTRYPOINT`。

### 提供数据和获取结果

训练任务很少是完全独立的；它们需要访问数据集并生成诸如训练模型、日志或评估指标等输出。如第 3 章所述，Docker 数据卷和绑定挂载是实现此目的的标准机制。

您可以通过 `docker run` 命令的 `-v` 或 `--mount` 标志来集成这些数据管理方法。

- **使用绑定挂载 (Bind Mounts)：** 适用于开发或数据直接位于主机上的情况。

  ```bash
  # 将本地 ./data 挂载到容器内的 /app/data
  # 将本地 ./output 挂载到容器内的 /app/output
  docker run \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/output:/app/output \
    your-ml-training-image:latest \
    python train.py --data-dir /app/data --model-dir /app/output/models
  ```

  在此示例中，`$(pwd)/data`（当前主机工作目录中的 `data` 目录）以只读方式（`:ro`）挂载到容器内部的 `/app/data`。`$(pwd)/output` 目录以读写方式挂载到 `/app/output`。容器内的训练脚本通过 `/app/data` 访问数据，并将模型保存到 `/app/output/models`，这些模型直接出现在主机上的 `./output/models` 目录中。
- **使用 Docker 数据卷 (Volumes)：** 推荐用于管理独立于主机文件系统结构的持久化数据。

  ```bash
  # 假设数据卷 'training_data_v1' 和 'model_artifacts_v1' 已存在
  docker run \
    -v training_data_v1:/app/data:ro \
    -v model_artifacts_v1:/app/output \
    your-ml-training-image:latest \
    python train.py --data-dir /app/data --model-dir /app/output/models
  ```

  这里，Docker 管理的数据卷被挂载到容器中。这可以将数据存储与主机的目录结构分离。

请记住，确保容器内部的路径（`/app/data`、`/app/output/models`）与您的训练脚本所预期的路径相符。

### 配置训练运行

如前所述，环境变量和命令行参数 (parameter)是配置训练任务的常见方式。`docker run` 为这两种方式提供了标志：

- **环境变量 (`-e` 或 `--env`)：** 适用于传递 API 密钥、学习率或标志等配置。

  ```bash
  docker run \
    -e LEARNING_RATE=0.005 \
    -e NUM_EPOCHS=30 \
    -e WANDB_API_KEY=your_secret_key \
    -v ... \
    your-ml-training-image:latest \
    python train.py 
    # 假设 train.py 从环境变量中读取 LEARNING_RATE 和 NUM_EPOCHS
  ```
- **命令行参数：** 直接在镜像名称之后（或在覆盖命令之后）传递。

docker run   
-v ...   
your-ml-training-image:latest   
python train.py --learning-rate 0.005 --epochs 30 --log-to-wandb
\`\`\`

它们之间的选择通常取决于训练脚本的设计。环境变量适用于需要在不同脚本执行中使用的密钥或设置，而命令行参数则明确且常用于特定运行参数，例如超参数 (hyperparameter)。

### 分离模式与交互模式执行

默认情况下，`docker run` 会将您的终端连接到容器的标准输入、输出和错误流。您将在终端中直接看到训练日志，并且命令提示符会一直阻塞，直到容器退出。这适用于短期任务或交互式调试。

对于长时间运行的训练任务，您通常会希望使用 `-d` 标志以**分离模式**运行容器：

```bash
docker run -d \
  --name long_training_run \
  -v training_data_v1:/app/data:ro \
  -v model_artifacts_v1:/app/output \
  -e LEARNING_RATE=0.001 \
  your-ml-training-image:latest \
  python train.py --data-dir /app/data --model-dir /app/output/models
```

此命令在后台启动容器并打印容器 ID。您的终端提示符会立即返回。

要查看分离模式下容器的日志，请使用 `docker logs` 命令：

```bash
# 实时跟踪日志
docker logs -f long_training_run

# 显示所有现有日志
docker logs long_training_run 
```

### 资源限制

机器学习 (machine learning)训练可以是计算密集型任务。`docker run` 允许您限制容器可以消耗的资源，防止单个任务独占主机资源。

- **CPU 限制 (`--cpus`)：** 指定容器可使用的 CPU 核心数量。
- **内存限制 (`--memory`)：** 设置最大 RAM 量。

```bash
docker run -d \
  --cpus="4" \
  --memory="16g" \
  --name resource_limited_training \
  -v ... \
  your-ml-training-image:latest \
  python train.py ... 
```

这可确保容器最多使用 4 个 CPU 核心和 16 GB RAM。

### 管理容器生命周期

- **命名容器 (`--name`)：** 分配一个易于记住的名称，可以更方便地管理容器（例如，查看日志、停止、删除），而不是依赖自动生成的 ID。

docker run -d --name my\_experiment\_run\_001 ...
docker logs my\_experiment\_run\_001
docker stop my\_experiment\_run\_001
\`\`\`

- **自动清理 (`--rm`)：** 对于通常只运行一次且完成后无需检查的训练任务，`--rm` 标志非常有用。当容器退出时，它会自动移除容器的文件系统。这可以防止您的系统被停止的容器占用。

  ```bash
  # 容器将在完成或出错时自动移除
  docker run --rm \
    --name transient_training \
    -v ... \
    your-ml-training-image:latest \
    python train.py ...
  ```

  注意：`--rm` 不能与 `-d` 一起使用。如果您在分离模式下运行任务，并希望在完成后将其移除，则需要使用 `docker rm <容器名称或 ID>` 手动移除它。

### `docker run` 训练过程可视化

`docker run` 命令协调多个组件，以在隔离环境中启动您的训练任务。

> 此图示说明了使用 `docker run` 执行训练任务时的流程。该命令启动该过程，指示 Docker 守护进程从镜像创建容器，挂载所需的数据卷或目录，注入环境变量，并执行指定的训练脚本。

通过熟练掌握 `docker run` 及其用于数据挂载、配置和生命周期管理的各种标志，您将获得对机器学习 (machine learning)训练任务执行方式的精确控制，显著提高了任务的一致性并简化了在不同机器或云环境中运行实验的流程。

## 参考资料

- [docker run](https://docs.docker.com/engine/reference/commandline/run/) — Docker Documentation (2024)
  Publisher: Docker
  涵盖 `docker run` 命令的基本用法，包括命令行参数、环境变量、资源分配和容器生命周期管理，有助于执行机器学习训练作业。
- [Use volumes](https://docs.docker.com/storage/volumes/) — Docker Documentation (2024)
  Publisher: Docker Inc.
  详细介绍了 Docker 卷和绑定挂载的原理与实际应用，对于在机器学习训练期间管理主机与容器之间的数据持久性和共享非常重要。
- [Docker Deep Dive](https://leanpub.com/dockerdeepdive) — Nigel Poulton (2023)
  Publisher: Nigel Poulton Ltd.
  一本全面的 Docker 指南，解释了容器化概念和实际操作。本书有助于理解在 Docker 容器中运行机器学习工作流的技术基础。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.barnesandnoble.com/w/building-machine-learning-powered-applications-emmanuel-ameisen/1136453123) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media
  提供了开发和部署机器学习应用程序的端到端实践指导，包括使用容器（如 Docker）实现机器学习模型和训练环境的复现性与高效部署的方法。
