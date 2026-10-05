# 第 2 章：Pydantic 数据处理与校验

来源：[原章节](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic)

[返回课程目录](../README.md)

构建 API 时，特别是针对机器学习模型，确保接收到的数据正确且结构化十分重要。“输入垃圾，输出垃圾”的原则直接适用；无效的输入数据可能导致预测错误或应用崩溃。FastAPI 使用 Pydantic 高效且声明式地处理数据校验。

本章主要讲解如何在 FastAPI 中使用 Pydantic 模型。您将学习如何：
*   使用 Pydantic 的类型注解定义数据模式。
*   自动校验传入的请求数据（例如 JSON 请求体）。
*   指定并校验传出的响应数据的格式。
*   处理和校验路径参数和查询参数。
*   应用约束并构建适合典型机器学习输入和输出的复杂数据模型。

学完本章后，您将能够创建能够可靠地处理数据的 API 端点，确保传递给机器学习模型的信息符合预期的格式和类型。

## 小节

- 1. [Pydantic 简介](01-Pydantic%20%E7%AE%80%E4%BB%8B.md)
- 2. [定义数据模型](02-%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E6%A8%A1%E5%9E%8B.md)
- 3. [请求体数据校验](03-%E8%AF%B7%E6%B1%82%E4%BD%93%E6%95%B0%E6%8D%AE%E6%A0%A1%E9%AA%8C.md)
- 4. [响应模型定义](04-%E5%93%8D%E5%BA%94%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89.md)
- 5. [处理路径参数和查询参数](05-%E5%A4%84%E7%90%86%E8%B7%AF%E5%BE%84%E5%8F%82%E6%95%B0%E5%92%8C%E6%9F%A5%E8%AF%A2%E5%8F%82%E6%95%B0.md)
- 6. [数据转换与约束](06-%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%8D%A2%E4%B8%8E%E7%BA%A6%E6%9D%9F.md)
- 7. [构建复杂数据模型](07-%E6%9E%84%E5%BB%BA%E5%A4%8D%E6%9D%82%E6%95%B0%E6%8D%AE%E6%A8%A1%E5%9E%8B.md)
- 8. [动手实践：验证机器学习输入数据](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic/quiz)
