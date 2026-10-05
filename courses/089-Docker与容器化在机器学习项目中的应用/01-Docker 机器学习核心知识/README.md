# 第 1 章：Docker 机器学习核心知识

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml)

[返回课程目录](../README.md)

机器学习开发常遇到环境配置、依赖冲突以及在不同系统上确保可复现性等问题。容器化，特别是使用 Docker，通过将应用程序及其依赖项打包在一起，提供了一种结构化的方法来解决这些问题。

本章将回顾机器学习工作流中 Docker 的基本要点，从而打下基础。我们将了解：

*   在机器学习项目中，使用容器实现一致性和可扩展性的具体优点。
*   Docker 核心组件：镜像、容器和层，以及它们与机器学习环境的关系。
*   管理容器生命周期的基本命令。
*   介绍如何使用 Dockerfile 定义自定义环境。
*   Docker 注册表（如 Docker Hub）在共享和获取镜像方面的作用。

在本章结束时，您将明白 Docker 对机器学习为何有益，并获得运行预构建机器学习镜像的实践经验。

## 小节

- 1. [为何容器化机器学习项目？](01-%E4%B8%BA%E4%BD%95%E5%AE%B9%E5%99%A8%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE%EF%BC%9F.md)
- 2. [Docker 镜像基本知识](02-Docker%20%E9%95%9C%E5%83%8F%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86.md)
- 3. [容器生命周期管理](03-%E5%AE%B9%E5%99%A8%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E7%AE%A1%E7%90%86.md)
- 4. [Dockerfile 简介](04-Dockerfile%20%E7%AE%80%E4%BB%8B.md)
- 5. [Docker 注册中心与仓库](05-Docker%20%E6%B3%A8%E5%86%8C%E4%B8%AD%E5%BF%83%E4%B8%8E%E4%BB%93%E5%BA%93.md)
- 6. [实践操作：运行预构建的机器学习镜像](06-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E8%BF%90%E8%A1%8C%E9%A2%84%E6%9E%84%E5%BB%BA%E7%9A%84%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%95%9C%E5%83%8F.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml/quiz)
