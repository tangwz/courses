---
course: "docker-for-ml-projects"
chapter: "building-ml-dockerfiles"
lesson: "docker-environment-variables"
sourceId: 5258
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/docker-environment-variables"
title: "使用环境变量"
description: "配置容器行为并使用环境变量传递设置。"
order: 5
plots: []
sourceHash: "537b4120de675830025c67ef324ff4601114be824d51d00224f874e88072a7f1"
sourceCorrections: []
---

环境变量提供了一种将配置设置传递到 Docker 容器中的标准机制。这种方法对于机器学习 (machine learning)项目尤其有价值，因为它允许动态配置容器行为，而无需修改底层代码或每次重新构建镜像。它们对于参数 (parameter)化机器学习环境、影响脚本执行以及管理敏感信息是必不可少的。

环境变量是动态命名值，可以影响运行进程在计算机上的行为方式。在 Docker 容器中，它们的功能类似，提供了一种从外部（Docker 主机或编排系统）向隔离的容器环境注入配置的方式。

### 使用 `ENV` 指令设置默认值

在 `Dockerfile` 中定义环境变量的主要方法是使用 `ENV` 指令。这会设置在镜像构建过程的后续步骤中以及从生成的镜像运行容器时都可用的变量。

`ENV` 指令有两种形式：

1. `ENV <key>=<value>`: 设置单个变量。如果值包含空格，请用引号括起来或用反斜杠转义空格。
2. `ENV <key1>=<value1> <key2>=<value2> ...`: 在单个指令中设置多个变量。这种形式通常更受青睐，因为它创建的镜像层较少。

让我们看一个在机器学习 (machine learning)背景下的例子：

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置默认日志级别和模型目录
ENV LOG_LEVEL=INFO \
    MODEL_DIR=/app/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 此脚本可能会使用 MODEL_DIR 变量
CMD ["python", "train.py"]
```

在此 `Dockerfile` 中，`LOG_LEVEL` 被设置为 `INFO`，`MODEL_DIR` 被设置为 `/app/models`。任何在容器内运行的脚本（例如 `train.py`）都可以访问这些值，通常使用编程语言中可用的标准库函数（例如，Python 中的 `os.environ.get('MODEL_DIR')`）。这样设置默认值可以使你的镜像更自给自足且可预期。

### 使用 `ARG` 的构建时参数 (parameter)

Docker 也提供了 `ARG` 指令，用于定义用户可以在构建时使用 `docker build` 命令的 `--build-arg` 标志传递的变量。

```dockerfile
# Dockerfile
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim

ARG CACHE_DATE=not_set
ENV LAST_REFRESHED_AT=$CACHE_DATE

# ... Dockerfile 的其余部分
```

在这里，`PYTHON_VERSION` 是一个构建参数，默认值为 `3.9`。你可以使用不同的 Python 版本来构建此镜像，如下所示：

```bash
docker build --build-arg PYTHON_VERSION=3.10 -t my-ml-app .
```

`ARG` 和 `ENV` 之间有一个重要的区别：

- `ARG` 变量通常只在镜像构建过程中可用。它们默认不会作为环境变量保留在最终镜像或运行中的容器中。
- `ENV` 变量在构建期间可用，并保留在最终镜像中，可供从该镜像启动的容器内运行的应用程序访问。

然而，你可以通过定义一个使用 `ARG` 变量的 `ENV` 指令，让构建参数作为环境变量保留，如上面示例中的 `LAST_REFRESHED_AT` 所示。这种模式对于将构建时信息嵌入 (embedding)到运行时环境很有用。

`ARG` 用于影响构建本身的参数（例如基础镜像版本、源仓库），或用于不想被烘焙到最终镜像层元数据中的临时构建秘密。`ENV` 用于应用程序所需的运行时配置默认值。

### 在运行时覆盖变量

环境变量的一个重要优点是能够在启动容器时覆盖 `Dockerfile` 中设置的默认值。这提供了灵活性，而无需重新构建镜像。`docker run` 命令为此接受 `-e`（或 `--env`）标志。

```bash
# 使用默认设置运行容器
docker run my-ml-app

# 覆盖 LOG_LEVEL 以进行调试
docker run -e LOG_LEVEL=DEBUG my-ml-app

# 覆盖模型目录
docker run -e MODEL_DIR=/data/production_model my-ml-app

# 设置 Dockerfile 中未定义的新变量
docker run -e API_KEY=xyz123 my-ml-app
```

你可以多次使用 `-e` 标志来设置或覆盖多个变量。这种运行时配置对于将通用机器学习 (machine learning)镜像调整适应特定任务、环境（开发、测试、生产）或数据集非常重要。

### 使用环境变量文件

通过命令行上的 `-e` 标志管理大量环境变量可能会变得繁琐。Docker 允许你将这些变量放在一个文件（通常命名为 `.env`）中，并在容器启动时使用 `--env-file` 标志传递该文件。

一个 `.env` 文件的例子：

```dotenv
# .env 文件
LOG_LEVEL=INFO
MODEL_DIR=/data/shared/models
DATABASE_URL=postgresql://user:pass@dbhost:5432/ml_results
S3_BUCKET=my-ml-artifacts
```

每行遵循 `KEY=VALUE` 格式。注释以 `#` 开头。

你可以像这样运行容器：

```bash
docker run --env-file .env my-ml-app
```

Docker 从 `.env` 文件读取变量并将其设置到容器环境中。如果存在冲突，使用 `-e` 明确设置的变量将覆盖 `.env` 文件中定义的变量。使用 `.env` 文件是管理配置的良好实践，特别是在不同部署环境之间分离设置时。它还有助于将敏感信息排除在 shell 历史记录或命令行日志之外。

### 机器学习 (machine learning)项目中的常见用途

环境变量广泛用于容器化机器学习工作流程中：

- **文件路径：** 指定输入数据集 (`DATA_DIR`)、输出模型 (`MODEL_DIR`)、日志 (`LOG_DIR`) 或临时文件 (`TMP_DIR`) 的位置。
- **超参数 (parameter) (hyperparameter)：** 虽然通常通过命令行参数传递给训练脚本以保持清晰，但较简单的参数（如 `LEARNING_RATE`、`BATCH_SIZE`）有时可以通过 `ENV` 设置，特别是对于推理 (inference)服务器。
- **凭证和 API 密钥：** 提供云存储 (AWS S3, Google Cloud Storage)、数据库或外部 API 的访问密钥。**重要安全提示：** 避免使用 `ENV` 直接将秘密嵌入 (embedding) `Dockerfile`。在生产环境中，优先使用 `-e`、`--env-file` 或专用秘密管理系统进行运行时注入。
- **配置标志：** 控制应用程序行为，例如设置 `LOG_LEVEL`、启用/禁用功能 (`ENABLE_GPU=true`)，或指定机器学习框架后端 (`KERAS_BACKEND=tensorflow`)。
- **网络设置：** 定义用于连接其他服务的主机名或端口（例如 `DATABASE_HOST=db`，`REDIS_PORT=6379`）。

### 最佳实践和安全

- **避免硬编码秘密：** 切勿使用 `ENV` 将密码或 API 密钥等敏感数据直接放入 `Dockerfile` 中。这些数据将成为镜像层的一部分，并且可以被检查。对于生产工作负载，请使用运行时注入（`-e`、`--env-file`）或适当的秘密管理工具。
- **设置合理默认值：** 在 `Dockerfile` 中使用 `ENV` 为常见配置提供合理的默认值。这使得镜像开箱即用更方便。
- **区分 `ARG` 和 `ENV`：** `ARG` 用于构建时定制，`ENV` 用于运行时配置。
- **文档化变量：** 清晰地记录你的镜像预期或支持的环境变量，包括它们的用途和可能的值。这通常在 `Dockerfile` 附带的 `README.md` 文件中完成。
- **优先运行时注入：** 对于经常更改或在不同环境之间有所不同的配置，请依赖运行时注入，而不是重新构建镜像。

环境变量是一种简单但有效的机制，用于配置你的容器化机器学习 (machine learning)应用程序。通过了解如何使用 `ENV`、`ARG`、`-e` 和 `--env-file`，你可以创建灵活、可重用的 Docker 镜像，能够适应各种部署场景，无需持续的代码更改或镜像重建。这提升了一致性并简化了你的机器学习环境管理。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker Documentation (2024)
  Publisher: Docker
  本指南涵盖所有Dockerfile指令，包括ENV和ARG，提供用法示例和详细信息。
- [docker run](https://docs.docker.com/engine/reference/commandline/run/) — Docker Documentation (2024)
  详细介绍运行Docker容器的方法，解释如何使用-e和--env-file等标志进行配置。
- [Best practices for building images](https://docs.docker.com/build/building/best-practices/) — Docker Documentation (2024)
  包含创建高效安全Docker镜像的指南，涵盖构建过程中敏感信息的处理。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本关于容器化的实用指南，说明如何在镜像构建和容器运行时使用环境变量进行配置。
