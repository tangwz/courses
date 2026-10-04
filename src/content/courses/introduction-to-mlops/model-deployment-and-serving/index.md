---
course: "introduction-to-mlops"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving"
sourceId: 1262
chapter: "model-deployment-and-serving"
title: "模型部署与服务"
order: 5
description: "学习模型部署的基础知识，包括使用 Docker 进行容器化、创建模型 API 以及理解部署模式。"
hasQuiz: false
---

训练好的机器学习模型只有在应用程序能用它进行预测时，才能产生价值。本章介绍将模型从本地环境迁移到生产系统，并使其提供预测服务所需的技术流程。我们将学习从保存模型文件到构建可运行、受监控服务的全过程。

你将学习如何：

*   使用 Docker 将模型及其依赖项打包到独立且可移植的环境中。
*   使用 Flask 等框架，通过 Web API 提供模型的预测功能。
*   区分在线（实时）和批处理（离线）部署模式。
*   了解模型注册表在管理生产模型及其版本中的作用。
*   掌握监控已部署模型的运行和性能问题的基础知识。

本章末尾设有动手实践环节，你将对一个简单模型进行容器化处理，并为部署做好准备。
