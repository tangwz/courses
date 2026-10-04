---
course: "docker-for-ml-projects"
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference"
sourceId: 981
chapter: "containerizing-ml-inference"
title: "机器学习模型推理的容器化"
order: 5
description: "学习如何将训练好的机器学习模型和推理代码打包成高效的 Docker 容器，使其适用于部署。"
hasQuiz: true
---

模型训练完成后，下一步就是准备模型以对新数据进行预测。本章主要讲解如何将训练好的机器学习模型和必要的推理代码打包到 Docker 容器中。目的是构建自包含、高效的单元，用于提供预测服务。

你将掌握以下实用方法：

*   在容器内设计和构建简单的推理服务，通常使用 Flask 或 FastAPI 等 Web 框架。
*   使用多阶段构建和精细的依赖管理等方法，优化生成的 Docker 镜像，以减小尺寸并提高效率。
*   配置用于推理的容器，包括暴露网络端口和实现健康检查，以确保服务正常运行。
*   在容器环境中有效打包模型和相关文件。

学完本章后，你将能够构建优化的 Docker 容器，用于为训练好的机器学习模型提供服务。
