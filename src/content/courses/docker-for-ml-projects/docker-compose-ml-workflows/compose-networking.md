---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "compose-networking"
sourceId: 5337
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/compose-networking"
title: "容器间网络连接"
description: "说明 Compose 如何设置网络，以使服务能够互相通信。"
order: 3
plots: []
sourceHash: "2339028f57dc6a3dc98e4d423f928fbc20a3fd477fc2cdbe6b0de531aa9e774a"
sourceCorrections: []
---

单独使用 `docker run` 管理多个容器时，要使它们之间进行通信，需要手动操作。你可能需要查找容器的IP地址（这些地址可能会变动），或者使用旧的链接机制。这个过程很繁琐，不利于定义可重现的应用栈。

Docker Compose 解决了这个问题，它会自动为你的应用程序配置网络。当你使用 `docker-compose up` 启动应用程序时，Compose 会在后台执行多个网络配置步骤：

1. **创建网络：** 默认情况下，Compose 会为你的应用程序创建一个专用的桥接网络（通常命名为 `<项目名称>_default`）。每个由 `docker-compose.yml` 文件定义的项目都会有自己独立的网络。
2. **连接服务：** 你的 `docker-compose.yml` 文件中定义的每个服务（容器）都会连接到这个默认网络。
3. **实现服务发现：** 同一 Compose 网络内的容器可以互相找到并通信，只需使用服务名作为主机名即可。

这种内置的服务发现功能是一个很大的优势。你无需担心容器的IP地址。如果你有一个名为 `db` 的服务和另一个名为 `api` 的服务，`api` 容器可以通过连接到主机名 `db` 的对应端口来访问 `db` 容器。

让我们来看一个简单的例子，其中一个 Python 应用程序 (`app`) 需要连接到 Redis 缓存 (`cache`)。

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: . # 假定当前目录中存在 Dockerfile
    ports:
      - "5000:5000" # 对外暴露端口 5000
    environment:
      # 应用程序可以使用 'cache' 作为主机名连接到 Redis
      REDIS_HOST: cache
    depends_on:
      - cache

  cache:
    image: redis:alpine
    # 内部通信不需要此处配置 'ports'
```

在这个例子中：

- Compose 创建了一个网络。
- 它使用本地的 `Dockerfile` 构建了 `app` 服务并启动了一个容器。
- 它拉取了 `redis:alpine` 镜像并启动了 `cache` 服务容器。
- `app` 和 `cache` 容器都连接到同一个由 Compose 管理的网络。
- `app` 容器可以使用主机名 `cache` 连接到 Redis 服务（正如 `REDIS_HOST: cache` 所指定的那样）。标准的 Redis 端口是 6379，因此应用程序代码中的连接字符串很可能会指向 `cache:6379`。
- `depends_on` 确保 `cache` 服务在 `app` 服务之前启动，但这不保证 Redis 在容器内部完全就绪，仅表示容器已启动。

> 该图显示了 `app` 服务容器如何通过自动创建的 Docker Compose 网络，使用服务名 `cache` 连接到 `cache` 服务容器。对 `app` 服务的外部访问通过主机的映射端口进行。

需要区分内部网络通信和外部端口暴露。

- **内部通信：** 同一 Compose 网络内的服务可以使用服务名和目标服务在*内部*监听的端口直接通信（例如，Redis 在其容器内部监听 6379 端口）。对于这种内部流量，你*不需要*在 `docker-compose.yml` 中使用 `ports` 指令。
- **外部访问：** `ports` 指令（例如 `app` 服务的 `"5000:5000"`）将主机上的端口映射到容器内部的端口。只有当你需要从 Docker 主机*外部*访问某个服务时，这才是必需的（例如，从你的网页浏览器或不受此 Compose 文件管理的其他应用程序）。在我们的例子中，Redis `cache` 服务不需要外部访问，因此它没有 `ports` 部分。

虽然默认网络对大多数应用来说已经足够，但 Docker Compose 也允许你定义和使用自定义网络，如果你需要对网络拓扑有更细致的控制，或者想连接不同 Compose 文件中的服务。然而，对于典型的机器学习 (machine learning)开发堆栈，如果其中包含 API、数据库或工作队列等组件，且这些组件都在单个 `docker-compose.yml` 中定义，默认网络为服务间通信提供了一个直接且实用的方案。这种自动网络配置和服务发现功能大大简化了定义和运行多容器应用程序的过程。

## 参考资料

- [Networking in Compose](https://docs.docker.com/compose/networking/) — Docker Documentation (2024)
  此官方文档全面介绍了Docker Compose如何自动设置和管理多容器应用程序的网络，涵盖服务发现和配置。
- [Understand Docker container networks](https://docs.docker.com/network/) — Docker Documentation (2024)
  Publisher: Docker Inc.
  这份官方指南解释了Docker中的基本网络概念，包括Docker Compose默认用于容器间通信的桥接网络驱动。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications
  本实用指南详细解释了Docker Compose，包括其网络功能、服务发现以及如何定义多容器应用程序以实现高效的开发和部署。
