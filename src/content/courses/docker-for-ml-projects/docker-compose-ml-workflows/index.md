---
course: "docker-for-ml-projects"
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows"
sourceId: 984
chapter: "docker-compose-ml-workflows"
title: "使用 Docker Compose 简化工作流程"
order: 6
description: "使用 Docker Compose 定义和管理多容器机器学习应用程序，用于本地开发和测试。"
hasQuiz: true
---

到目前为止，我们一直专注于构建和运行独立的 Docker 容器，用于训练或推理等特定任务。然而，许多机器学习的开发和部署场景涉及多个相互配合的组件。例如，您可能需要一个推理 API 容器、一个用于存储元数据的数据库容器，甚至是一个用于异步任务的消息队列容器。使用独立的 `docker run` 命令来管理这些独立容器的设置、网络配置和生命周期，可能会变得效率低下且容易出错。

本章将介绍 Docker Compose，这是一个专门设计的工具，用于简化多容器 Docker 应用程序的定义和管理。您将学习如何使用一个单一的配置文件（通常命名为 `docker-compose.yml`）来定义构成您应用程序堆栈的所有服务。我们将讲解服务的定义、为容器间通信建立网络、使用卷管理持久数据、使用环境变量配置服务，以及直接在 Compose 工作流程中构建容器镜像。在本章结束时，您将能够使用 Docker Compose 在本地搭建和运行复杂的机器学习开发环境。
