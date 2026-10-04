---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "compose-ml-stack-examples"
sourceId: 5349
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/compose-ml-stack-examples"
title: "常见机器学习栈示例 (例如，API + 数据库)"
description: "查看如何定义常见的机器学习开发栈示例，例如带有支持数据库的推理 API。"
order: 7
plots: []
sourceHash: "79ac8a940d9c11c291f504363f4ad892d3fa733d0d2ba72db7b1bd6a6ba06355"
sourceCorrections: []
---

Docker Compose 中的独立服务、卷和网络是核心组成部分。这些元素结合起来，构成机器学习 (machine learning)项目中常见的应用栈。使用 `docker-compose.yml` 文件，可以声明式地高效定义和管理这些多组件系统，用于本地开发和测试，从而模拟更复杂的部署配置。

### 示例 1: 带数据库后端的推理 (inference) API

一个常见模式是部署已训练模型作为 API 端点，用于进行预测。通常，此 API 需要与数据库交互，以存储预测日志、获取用户信息或管理其他应用状态。Docker Compose 使设置这种 API 加数据库的组合变得简单直接。

考虑一个由以下部分组成的应用：

1. **API 服务 (`api`):** 一个自定义容器，运行一个网络框架（如 Flask 或 FastAPI），加载训练好的模型并提供预测端点。它需要访问数据库。
2. **数据库服务 (`db`):** 一个标准数据库容器（如 PostgreSQL 或 MySQL），API 服务可以在此容器中读写数据。它的数据需要在容器重启后保持不变。

> 此图表显示用户请求到达 API 服务，该服务通过共享的 Docker 网络与数据库服务通信。数据库服务使用卷来持久化数据。

以下是表示此栈的简化 `docker-compose.yml`：

```yaml
version: '3.8' # 指定 Compose 文件版本

services:
  api:
    build: ./api # API Dockerfile 所在目录的路径
    ports:
      - "5000:5000" # 将主机端口 5000 映射到容器端口 5000
    volumes:
      - ./api:/app # 将本地 API 代码挂载到容器中 (用于开发)
      - ./models:/app/models # 挂载模型目录
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydatabase
      # 其他 API 特定配置...
    depends_on:
      - db # 等待数据库服务健康 (如果定义了健康检查)
    networks:
      - app-net

  db:
    image: postgres:14-alpine # 使用官方 PostgreSQL 镜像
    volumes:
      - postgres_data:/var/lib/postgresql/data # 挂载命名卷以持久化数据
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydatabase
    networks:
      - app-net
    # 可选: 为 depends_on 添加健康检查
    # healthcheck:
    #   test: ["CMD-SHELL", "pg_isready -U user -d mydatabase"]
    #   interval: 10s
    #   timeout: 5s
    #   retries: 5

volumes:
  postgres_data: # 定义命名卷

networks:
  app-net: # 定义自定义网络
    driver: bridge
```

此示例的要点总结：

- **服务：** 我们定义了两个服务：`api` 和 `db`。
- **构建与镜像：** `api` 服务使用 `build: ./api` 从本地 `Dockerfile` 构建，而 `db` 服务使用来自 Docker Hub 的预构建镜像 (`image: postgres:14-alpine`)。
- **网络：** 两个服务都连接到一个自定义网络 `app-net`。这使得 `api` 服务可以使用主机名 `db`（服务名称）和标准 PostgreSQL 端口 `5432` 连接到数据库。连接字符串 `postgresql://user:password@db:5432/mydatabase` 很好地说明了这一点。
- **卷：** `api` 服务使用了绑定挂载 (`./api:/app`)，以便在开发过程中主机上的代码更改可以立即反映在容器中。`db` 服务使用命名卷 (`postgres_data`)，以确保即使 `db` 容器被移除并重新创建，数据库文件也能持久存在。
- **环境变量：** `environment` 用于传递配置。`db` 服务使用它们来初始化数据库（用户、密码、名称）。`api` 服务使用它们来知道如何连接到数据库。
- **端口：** `ports` 映射 `- "5000:5000"` 使 API 可以通过主机上的端口 5000 访问。
- **`depends_on`：** 这确保了 `db` 服务在 `api` 服务尝试连接之前启动，尽管如果没有 `healthcheck`，它并不能保证容器*内部*的数据库已完全准备就绪。

在包含此文件的目录中运行 `docker compose up` 将会构建 API 镜像（如果需要）、拉取 PostgreSQL 镜像、创建网络和卷，并启动两个容器。

### 示例 2: 带 MLflow 跟踪的训练任务

另一个常见情况是在容器中运行训练脚本，同时将指标、参数 (parameter)和工件记录到像 MLflow 这样的实验跟踪服务器。Docker Compose 可以管理训练容器、MLflow 跟踪服务器，以及可能用于 MLflow 的后端数据库。

组件：

1. **训练服务 (`trainer`):** 一个运行机器学习 (machine learning)训练脚本的容器。它需要知道 MLflow 服务器的 URI。
2. **MLflow 服务器服务 (`mlflow`):** 运行 MLflow 跟踪服务器的用户界面和 API。
3. **后端存储服务 (`db`):** （可选，但建议为求稳定）一个数据库（例如 PostgreSQL），由 MLflow 用于存储实验元数据。

> 此图表显示训练服务通过网络向 MLflow 服务器发送日志。MLflow 服务器将元数据存储在后端数据库中，并从共享卷/挂载点读写工件。开发者可访问 MLflow 用户界面。

一个可能的 `docker-compose.yml` 可能如下所示：

```yaml
version: '3.8'

srvices:
  trainer:
    build: ./training # 训练脚本 Dockerfile 上下文路径
    command: python train.py --data /data/input.csv --model-output /artifacts
    volumes:
      - ./training/src:/app # 挂载训练代码
      - ./data:/data # 挂载输入数据
      - mlflow_artifacts:/artifacts # 挂载工件卷
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5001 # 告知脚本 MLflow 服务器位置
    depends_on:
      mlflow:
        condition: service_started # 基本依赖
      db:
        condition: service_healthy # 等待数据库 (需要健康检查)
    networks:
      - ml-net

  mlflow:
    # 考虑使用官方或维护良好的 MLflow 镜像，
    # 或在需要自定义时自行构建。
    # 此示例假设存在一个接受这些参数的预构建镜像。
    image: ghcr.io/mlflow/mlflow:v2.10.0
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow_user:mlflow_pass@db:5432/mlflow_db
      --default-artifact-root /mlflow_artifacts
      --host 0.0.0.0
      --port 5001
    ports:
      - "5001:5001" # 暴露 MLflow UI 端口
    volumes:
      - mlflow_artifacts:/mlflow_artifacts # 共享工件卷
    depends_on:
      db:
        condition: service_healthy # 等待数据库
    networks:
      - ml-net

  db:
    image: postgres:14-alpine
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=mlflow_user
      - POSTGRES_PASSWORD=mlflow_pass
      - POSTGRES_DB=mlflow_db
    networks:
      - ml-net
    healthcheck: # 此项对于 depends_on condition: service_healthy 必不可少
      test: ["CMD-SHELL", "pg_isready -U mlflow_user -d mlflow_db"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  mlflow_db_data:
  mlflow_artifacts: # 用于存储模型工件、参数等的卷。

networks:
  ml-net:
    driver: bridge
```

在此配置中：

- `trainer` 服务运行训练脚本 (`train.py`)。它通过 `MLFLOW_TRACKING_URI` 环境变量接收 MLflow 服务器的位置 (`http://mlflow:5001`)。脚本中的 MLflow 客户端库使用此 URI 进行连接。
- `mlflow` 服务运行跟踪服务器。其 `command` 将 PostgreSQL 数据库（`db` 服务）指定为 `--backend-store-uri`，并将共享卷（`mlflow_artifacts`）指定为 `--default-artifact-root`。
- `db` 服务提供 PostgreSQL 数据库用于 MLflow 元数据持久化，使用其自身的命名卷 (`mlflow_db_data`)。增加了 `healthcheck`，以便其他服务可以使用 `depends_on` 稳定地等待它就绪。
- 共享命名卷 `mlflow_artifacts` 被 `trainer`（用于写入工件）和 `mlflow`（用于读取/提供工件）都挂载。
- 带有 `condition: service_healthy` 的 `depends_on` 确保 `trainer` 和 `mlflow` 服务等待 `db` 就绪。

这些示例阐明了 Docker Compose 如何定义机器学习开发中常见的互联服务。你可以通过添加数据预处理服务、用于异步任务的消息队列（如 Redis 或 RabbitMQ）、监控工具或不同类型的数据库来扩展这些模式，所有这些都可以在单个 `docker-compose.yml` 文件中管理。这种方式显著简化了本地开发环境的设置和管理，使其与潜在的生产部署结构非常接近。

## 参考资料

- [Overview of Docker Compose](https://docs.docker.com/compose/) — Docker (2024)
  Publisher: Docker
  Docker官方文档，提供了使用Docker Compose声明式定义和管理多容器应用的关键信息，包括本节中展示的服务、网络和卷。
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — MLflow Team (2024)
  Publisher: MLflow (LF AI & Data Foundation)
  MLflow官方文档，详细介绍了MLflow跟踪服务器的设置和使用，包括后端和工件存储，与实验跟踪示例直接相关。
- [Practical MLOps: How to Take a Machine Learning Model from Idea to Production](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Vishnu Rachakonda (2022)
  Publisher: O'Reilly Media
  一本关于在生产环境中构建和部署机器学习系统的综合资源，提供了架构模式、MLOps最佳实践以及容器化ML堆栈的背景知识。
