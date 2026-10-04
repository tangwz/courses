---
course: "basics-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker"
sourceId: 802
chapter: "intro-containerization-docker"
title: "Docker 容器化简介"
order: 4
description: "学习 Docker 基本知识，将你的机器学习应用程序及其依赖项打包成可移植容器。"
hasQuiz: true
---

你现在已经构建了一个简单的 Web 服务，能够从你训练好的模型中提供预测结果。下一个难题是确保这个服务运行的一致性，无论它部署在哪里——无论是你的本地机器、队友的电脑还是生产服务器。操作系统、已安装库和配置的差异常常导致一个常见问题：“在我的机器上能运行！”

本章介绍容器化，这是一种将你的应用程序及其所有依赖项、库和配置文件打包成一个标准化单元的方法。我们将专注于 Docker，这是一个广泛用于构建和运行这些容器的平台。

你将学到：

*   容器化的基本原理。
*   Docker 的基本知识，包括镜像和容器。
*   如何编写 `Dockerfile` 来定义应用程序的运行环境。
*   为先前开发的 Flask 预测服务构建 Docker 镜像的步骤。
*   如何在 Docker 容器中运行你的应用程序。

学完本章，你将能够把你的简单机器学习预测服务打包成一个可移植的 Docker 容器。
