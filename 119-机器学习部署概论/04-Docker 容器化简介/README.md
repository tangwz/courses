# 第 4 章：Docker 容器化简介

来源：[原章节](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker)

[返回课程目录](../README.md)

你现在已经构建了一个简单的 Web 服务，能够从你训练好的模型中提供预测结果。下一个难题是确保这个服务运行的一致性，无论它部署在哪里——无论是你的本地机器、队友的电脑还是生产服务器。操作系统、已安装库和配置的差异常常导致一个常见问题：“在我的机器上能运行！”

本章介绍容器化，这是一种将你的应用程序及其所有依赖项、库和配置文件打包成一个标准化单元的方法。我们将专注于 Docker，这是一个广泛用于构建和运行这些容器的平台。

你将学到：

*   容器化的基本原理。
*   Docker 的基本知识，包括镜像和容器。
*   如何编写 `Dockerfile` 来定义应用程序的运行环境。
*   为先前开发的 Flask 预测服务构建 Docker 镜像的步骤。
*   如何在 Docker 容器中运行你的应用程序。

学完本章，你将能够把你的简单机器学习预测服务打包成一个可移植的 Docker 容器。

## 小节

- 1. [什么是容器化？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%AE%B9%E5%99%A8%E5%8C%96%EF%BC%9F.md)
- 2. [Docker 入门](02-Docker%20%E5%85%A5%E9%97%A8.md)
- 3. [Docker 核心理念：镜像与容器](03-Docker%20%E6%A0%B8%E5%BF%83%E7%90%86%E5%BF%B5%EF%BC%9A%E9%95%9C%E5%83%8F%E4%B8%8E%E5%AE%B9%E5%99%A8.md)
- 4. [安装 Docker](04-%E5%AE%89%E8%A3%85%20Docker.md)
- 5. [编写一个简单的Dockerfile](05-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Dockerfile.md)
- 6. [为Flask应用构建Docker镜像](06-%E4%B8%BAFlask%E5%BA%94%E7%94%A8%E6%9E%84%E5%BB%BADocker%E9%95%9C%E5%83%8F.md)
- 7. [在 Docker 容器中运行应用](07-%E5%9C%A8%20Docker%20%E5%AE%B9%E5%99%A8%E4%B8%AD%E8%BF%90%E8%A1%8C%E5%BA%94%E7%94%A8.md)
- 8. [动手实践：容器化预测服务](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1.md)

章节测验：[在线测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/quiz)
