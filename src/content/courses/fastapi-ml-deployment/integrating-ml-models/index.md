---
course: "fastapi-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models"
sourceId: 975
chapter: "integrating-ml-models"
title: "整合机器学习模型"
order: 3
description: "学习通过 FastAPI 预测端点序列化、加载并提供机器学习模型服务的方法。"
hasQuiz: true
---

在明确了如何使用 Pydantic 定义和验证数据结构后，下一步就是将机器学习模型本身集成到你的 FastAPI 应用程序中。本章将介绍如何连接已训练模型与实时 API 端点的具体实践。

你将学习以下方法：
*   将训练好的模型序列化以便存储，并反序列化以便使用。
*   在 FastAPI 应用程序生命周期内高效加载模型文件。
*   创建专门用于接收输入数据并返回模型预测结果的 API 端点。
*   管理与你的应用程序相关的模型文件（artifacts）。
*   使用 FastAPI 的依赖注入机制为预测逻辑提供模型。

在本章结束时，你将能够构建可用的 API 端点，从你训练的机器学习模型提供预测服务。
