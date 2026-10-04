# 第 3 章：使用 Flask 创建预测服务

来源：[原章节](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask)

[返回课程目录](../README.md)

训练好的模型保存后，下一步是让它投入使用，以便接收输入数据并返回预测结果。本章将介绍如何通过网络提供模型的预测功能。

我们将介绍应用程序接口（API），并解释它们在模型服务中的作用。你将学习 Flask Web 框架的基础知识，并使用它来构建一个基本的 Web 服务。该服务将加载一个序列化的模型，并通过 HTTP 处理预测请求。主要内容包括配置 Flask、定义用于预测的路由、处理输入数据（通常是 JSON 格式）以及构建预测响应。学完本章后，你将能够构建一个简单且可用的 API 端点，能够从你的机器学习模型提供预测结果。

## 小节

- 1. [什么是API？](01-%E4%BB%80%E4%B9%88%E6%98%AFAPI%EF%BC%9F.md)
- 2. [Web框架简介](02-Web%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B.md)
- 3. [设置 Flask](03-%E8%AE%BE%E7%BD%AE%20Flask.md)
- 4. [构建一个基础的 Flask 应用](04-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80%E7%9A%84%20Flask%20%E5%BA%94%E7%94%A8.md)
- 5. [在 Flask 中加载已保存的模型](05-%E5%9C%A8%20Flask%20%E4%B8%AD%E5%8A%A0%E8%BD%BD%E5%B7%B2%E4%BF%9D%E5%AD%98%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 6. [定义预测端点](06-%E5%AE%9A%E4%B9%89%E9%A2%84%E6%B5%8B%E7%AB%AF%E7%82%B9.md)
- 7. [处理输入数据 (JSON)](07-%E5%A4%84%E7%90%86%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE%20%28JSON%29.md)
- 8. [返回预测结果](08-%E8%BF%94%E5%9B%9E%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
- 9. [在本地测试你的 API](09-%E5%9C%A8%E6%9C%AC%E5%9C%B0%E6%B5%8B%E8%AF%95%E4%BD%A0%E7%9A%84%20API.md)
- 10. [动手实践：构建一个简单的Flask预测API](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Flask%E9%A2%84%E6%B5%8BAPI.md)

章节测验：[在线测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/quiz)
