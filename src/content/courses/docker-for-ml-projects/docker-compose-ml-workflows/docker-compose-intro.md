---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "docker-compose-intro"
sourceId: 5333
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/docker-compose-intro"
title: "Docker Compose 简介"
description: "理解 Docker Compose 的作用及其基本语法 (`docker-compose.yml`)。"
order: 1
plots: []
sourceHash: "2865a663b539f68350a50e0ff5b5bf8e7a7c367a771ef5bbf360b90ae8a982ca"
sourceCorrections: []
---

使用单独的 `docker run` 命令管理关联服务（例如 API、数据库或处理工作器）很快就会变得繁琐。您需要手动配置网络以使它们能够通信，管理启动顺序，一致地处理端口映射，并且每次启动或停止应用程序环境时都重复这些步骤。这种手动过程不仅乏味，而且容易出错，特别是随着服务数量的增加。

Docker Compose 提供了一种更具结构化且高效的方法。它是一个专门为定义和运行多容器 Docker 应用程序而构建的工具。您无需发出带有复杂参数 (parameter)的多个 `docker run` 命令，而是将您的整个应用程序堆栈（包括所有服务、网络和卷）都在一个配置文件中描述。

这个配置文件，通常命名为 `docker-compose.yml`，使用 YAML（YAML Ain't Markup Language）格式，该格式旨在易于人类阅读。在此文件中，您以声明方式定义应用程序的构成。您指定每个服务应使用的镜像（例如您的 ML 推理 (inference) API 或 PostgreSQL 数据库）、它们应如何联网、应暴露哪些端口以及数据应如何使用卷持久化。

使用 Docker Compose 的主要优点是简化操作。通过一个命令，通常是 `docker-compose up`，Compose 会读取您的 `docker-compose.yml` 文件，并负责创建（并可选地构建）必要的镜像、设置网络、创建卷，以及以正确的顺序和配置启动所有已定义的容器。同样，`docker-compose down` 会停止并移除与您的应用程序堆栈相关的容器、网络和卷。

> 图示从多个 `docker run` 命令到单个 `docker-compose up` 命令管理在 `docker-compose.yml` 中定义的应用程序的转变。

虽然 Docker Compose 可以在多种环境下使用，但它对于为多组件机器学习 (machine learning)应用程序设置本地开发和测试环境特别有效。它使开发人员能够快速启动工作所需的整个堆栈，确保不同机器之间的一致性。在接下来的章节中，我们将查看 `docker-compose.yml` 文件的结构，并学习如何为常见的机器学习工作流定义服务、网络和卷。

## 参考资料

- [Overview of Docker Compose](https://docs.docker.com/compose/overview/) — Docker Documentation (Accessed 2023)
  这份官方指南介绍了Docker Compose在定义和运行多容器应用程序方面的作用。
- [Get started with Docker Compose](https://docs.docker.com/compose/gettingstarted/) — Docker Documentation (Accessed 2023)
  提供使用Docker Compose设置多服务应用程序的实用步骤和示例。
