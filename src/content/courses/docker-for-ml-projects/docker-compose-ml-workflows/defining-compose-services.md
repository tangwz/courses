---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "defining-compose-services"
sourceId: 5335
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/defining-compose-services"
title: "在`docker-compose.yml`中定义服务"
description: "了解如何定义数据库、API 或工作器等单独的容器（服务）。"
order: 2
plots: []
sourceHash: "78e7243c043f0dddc2908e001c84e2612b2dd020b4c3177fab16366d5a895066"
sourceCorrections: []
---

Docker Compose 配置的核心是 `docker-compose.yml` 文件中的 `services` 顶级元素。该元素用于定义构成您应用程序的每个独立组件或“服务”。可以将服务视为多容器设置中容器的蓝图。每个服务通常会在自己的容器中运行，它们彼此独立，但能够与同一文件中定义的其他服务通信，这一主题稍后会详细介绍。

让我们考虑机器学习 (machine learning)开发中的一个常见情况：您可能有一个 Web 应用程序（可能是提供预测的 Flask 或 FastAPI API）和一个单独的数据库服务（如 PostgreSQL 或 Redis），用于存储模型元数据或用户信息。在 Docker Compose 中，这些都将在 `services` 键下定义为不同的服务。

您给每个服务起的名字（例如 `api`、`db`、`worker`）很重要。它会成为 Compose 创建的 Docker 网络中的一个主机名，使服务能够轻松地互相发现和通信。

要定义一个服务，您首先要指定其名称，然后是其配置细节。最基本的配置部分是告诉 Compose 该服务要使用哪个容器镜像。有两种主要方式可以做到这一点：

1. **使用预构建镜像 (`image`)**：如果您的服务依赖于标准、公开可用的镜像（例如官方 Python 镜像、像 `postgres` 这样的数据库镜像，或 Docker Hub 上的预构建机器学习框架镜像），您可以使用 `image` 指令。您会指定镜像名称和标签，就像使用 `docker pull` 或 `docker run` 一样。

   ```yaml
   # docker-compose.yml
   version: '3.8' # 指定 Compose 文件版本（可选但推荐）

   services:
     database:
       image: postgres:14-alpine # 使用官方 PostgreSQL 14 Alpine 镜像
       # ... 数据库服务的其他配置 ...

     redis_cache:
       image: redis:7-alpine # 使用官方 Redis 7 Alpine 镜像
       # ... Redis 服务的其他配置 ...
   ```
2. **从 Dockerfile 构建 (`build`)**：如果您的服务需要自定义环境，例如您的特定机器学习应用程序代码、`requirements.txt` 中定义的依赖项或特定配置，您通常会从 Dockerfile 构建镜像。`build` 指令会告诉 Compose 在何处查找构建上下文 (context)（包含 Dockerfile 和构建所需任何文件的目录）。

   - 如果 Dockerfile 命名为 `Dockerfile` 且与 `docker-compose.yml` 位于同一目录中，您只需指定上下文路径：

     ```yaml
     # docker-compose.yml
     version: '3.8'

     services:
       ml_api:
         build: . # 使用当前目录中的 Dockerfile 构建镜像
         # ... API 服务的其他配置 ...
     ```
   - 如果您的项目结构更复杂，或者您的 Dockerfile 有不同的名称，您可以在 `build` 指令中提供更多细节：

     ```yaml
     # docker-compose.yml
     version: '3.8'

     services:
       training_job:
         build:
           context: ./training_module # 包含 Dockerfile 和源代码的目录路径
           dockerfile: Dockerfile.train # 指定自定义 Dockerfile 名称
         # ... 训练服务的其他配置 ...
     ```

在单个 `docker-compose.yml` 文件中混合使用这些方法是很常见的。您可以使用 `build` 构建您的自定义 API 服务，同时使用 `image` 拉取标准数据库镜像。

这是一个组合示例：

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 机器学习预测 API 的服务
  prediction_api:
    build: ./api # 假设 ./api/Dockerfile 存在
    # 端口、卷、环境变量等配置稍后将在此处添加

  # 数据库服务
  metadata_db:
    image: postgres:14-alpine
    # 卷、数据库设置的环境变量等配置稍后将在此处添加
```

这个简单结构定义了两个服务：`prediction_api`，它将从 `./api` 目录中的 Dockerfile 构建；以及 `metadata_db`，它将使用预构建的 `postgres:14-alpine` 镜像。

> `docker-compose.yml` 文件定义了服务，指定是根据本地 Dockerfile 构建服务（`prediction_api`），还是拉取现有镜像（`metadata_db`）。

尽管指定镜像来源（`image` 或 `build`）是基本的，但定义服务涉及更多内容。在后续部分中，我们将讨论如何在 `docker-compose.yml` 文件中，在每个服务定义下，配置这些服务之间的网络、使用卷管理持久化数据、暴露端口、设置环境变量，以及控制服务之间的启动顺序和依赖关系。

## 参考资料

- [Compose file | Docker Docs](https://docs.docker.com/compose/compose-file/) — Docker (2024)
  Publisher: Docker
  Docker Compose文件语法和结构的官方指南，包含服务定义以及'image'和'build'指令。
- [Dockerfile reference | Docker Docs](https://docs.docker.com/engine/reference/builder/) — Docker (2024)
  Publisher: Docker
  Dockerfile语法和指令的参考文献，这些指令在用'build'指令定义服务时使用。
- [Docker in Practice](https://www.manning.com/books/docker-in-practice-second-edition) — Ian Miell, Aidan Hobson Sayers (2019)
  Publisher: Manning Publications; Pages: 384
  一本实用书籍，提供使用Docker的示例和解释，包含使用Compose的多容器应用。
- [Docker: Up & Running: Shipping Reliable Containers in Production](https://www.oreilly.com/library/view/docker-up-running/9781098131821/) — Karl Matthias, Sean P. Kane (2023)
  Publisher: O'Reilly Media; Pages: 418
  本书涵盖Docker基础概念、镜像创建以及如何使用Docker Compose配置多容器应用。
