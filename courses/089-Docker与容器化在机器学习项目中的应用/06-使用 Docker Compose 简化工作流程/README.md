# 第 6 章：使用 Docker Compose 简化工作流程

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows)

[返回课程目录](../README.md)

到目前为止，我们一直专注于构建和运行独立的 Docker 容器，用于训练或推理等特定任务。然而，许多机器学习的开发和部署场景涉及多个相互配合的组件。例如，您可能需要一个推理 API 容器、一个用于存储元数据的数据库容器，甚至是一个用于异步任务的消息队列容器。使用独立的 `docker run` 命令来管理这些独立容器的设置、网络配置和生命周期，可能会变得效率低下且容易出错。

本章将介绍 Docker Compose，这是一个专门设计的工具，用于简化多容器 Docker 应用程序的定义和管理。您将学习如何使用一个单一的配置文件（通常命名为 `docker-compose.yml`）来定义构成您应用程序堆栈的所有服务。我们将讲解服务的定义、为容器间通信建立网络、使用卷管理持久数据、使用环境变量配置服务，以及直接在 Compose 工作流程中构建容器镜像。在本章结束时，您将能够使用 Docker Compose 在本地搭建和运行复杂的机器学习开发环境。

## 小节

- 1. [Docker Compose 简介](01-Docker%20Compose%20%E7%AE%80%E4%BB%8B.md)
- 2. [在\`docker-compose.yml\`中定义服务](02-%E5%9C%A8%60docker-compose.yml%60%E4%B8%AD%E5%AE%9A%E4%B9%89%E6%9C%8D%E5%8A%A1.md)
- 3. [容器间网络连接](03-%E5%AE%B9%E5%99%A8%E9%97%B4%E7%BD%91%E7%BB%9C%E8%BF%9E%E6%8E%A5.md)
- 4. [在 Compose 中使用卷](04-%E5%9C%A8%20Compose%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E5%8D%B7.md)
- 5. [Compose 中的环境变量](05-Compose%20%E4%B8%AD%E7%9A%84%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F.md)
- 6. [使用 Compose 构建镜像](06-%E4%BD%BF%E7%94%A8%20Compose%20%E6%9E%84%E5%BB%BA%E9%95%9C%E5%83%8F.md)
- 7. [常见机器学习栈示例 (例如，API + 数据库)](07-%E5%B8%B8%E8%A7%81%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A0%88%E7%A4%BA%E4%BE%8B%20%28%E4%BE%8B%E5%A6%82%EF%BC%8CAPI%20%2B%20%E6%95%B0%E6%8D%AE%E5%BA%93%29.md)
- 8. [动手练习：使用 Compose 构建机器学习应用](08-%E5%8A%A8%E6%89%8B%E7%BB%83%E4%B9%A0%EF%BC%9A%E4%BD%BF%E7%94%A8%20Compose%20%E6%9E%84%E5%BB%BA%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/quiz)
