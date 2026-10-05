# 第 3 章：整合机器学习模型

来源：[原章节](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models)

[返回课程目录](../README.md)

在明确了如何使用 Pydantic 定义和验证数据结构后，下一步就是将机器学习模型本身集成到你的 FastAPI 应用程序中。本章将介绍如何连接已训练模型与实时 API 端点的具体实践。

你将学习以下方法：
*   将训练好的模型序列化以便存储，并反序列化以便使用。
*   在 FastAPI 应用程序生命周期内高效加载模型文件。
*   创建专门用于接收输入数据并返回模型预测结果的 API 端点。
*   管理与你的应用程序相关的模型文件（artifacts）。
*   使用 FastAPI 的依赖注入机制为预测逻辑提供模型。

在本章结束时，你将能够构建可用的 API 端点，从你训练的机器学习模型提供预测服务。

## 小节

- 1. [机器学习模型的序列化与反序列化](01-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%BA%8F%E5%88%97%E5%8C%96%E4%B8%8E%E5%8F%8D%E5%BA%8F%E5%88%97%E5%8C%96.md)
- 2. [将模型载入FastAPI应用](02-%E5%B0%86%E6%A8%A1%E5%9E%8B%E8%BD%BD%E5%85%A5FastAPI%E5%BA%94%E7%94%A8.md)
- 3. [创建预测端点](03-%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E7%AB%AF%E7%82%B9.md)
- 4. [处理不同输入格式](04-%E5%A4%84%E7%90%86%E4%B8%8D%E5%90%8C%E8%BE%93%E5%85%A5%E6%A0%BC%E5%BC%8F.md)
- 5. [返回预测结果和概率](05-%E8%BF%94%E5%9B%9E%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C%E5%92%8C%E6%A6%82%E7%8E%87.md)
- 6. [模型工件管理](06-%E6%A8%A1%E5%9E%8B%E5%B7%A5%E4%BB%B6%E7%AE%A1%E7%90%86.md)
- 7. [模型加载的依赖注入](07-%E6%A8%A1%E5%9E%8B%E5%8A%A0%E8%BD%BD%E7%9A%84%E4%BE%9D%E8%B5%96%E6%B3%A8%E5%85%A5.md)
- 8. [实践：构建模型预测服务](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1.md)

章节测验：[在线测验](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/quiz)
