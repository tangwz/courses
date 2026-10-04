---
course: "basics-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask"
sourceId: 798
chapter: "prediction-service-flask"
title: "使用 Flask 创建预测服务"
order: 3
description: "学习如何使用 Flask 框架将你保存的机器学习模型包装成一个简单的 Web API，并通过 HTTP 提供预测结果。"
hasQuiz: true
---

训练好的模型保存后，下一步是让它投入使用，以便接收输入数据并返回预测结果。本章将介绍如何通过网络提供模型的预测功能。

我们将介绍应用程序接口（API），并解释它们在模型服务中的作用。你将学习 Flask Web 框架的基础知识，并使用它来构建一个基本的 Web 服务。该服务将加载一个序列化的模型，并通过 HTTP 处理预测请求。主要内容包括配置 Flask、定义用于预测的路由、处理输入数据（通常是 JSON 格式）以及构建预测响应。学完本章后，你将能够构建一个简单且可用的 API 端点，能够从你的机器学习模型提供预测结果。
