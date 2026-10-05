# 第 4 章：组织和测试 FastAPI 应用

来源：[原章节](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi)

[返回课程目录](../README.md)

在学习了如何创建端点、使用 Pydantic 处理数据以及集成机器学习模型之后，我们现在将注意力转向构建更易于维护和可靠的应用程序。随着您的机器学习 API 变得越来越复杂，适当的组织和测试对于项目的持续性变得重要。

本章提供关于如何有效地组织 FastAPI 项目和实施测试策略的指导。你将学习如何使用 `APIRouter` 来模块化代码，应用关注点分离的原则，管理项目依赖，以及使用 FastAPI 的 `TestClient` 编写单元测试和集成测试。我们还将介绍日志记录实践以及处理应用程序配置和秘密的方法。目标是让你掌握开发可扩展和可靠的机器学习部署服务所需的技术。

## 小节

- 1. [使用路由组织项目](01-%E4%BD%BF%E7%94%A8%E8%B7%AF%E7%94%B1%E7%BB%84%E7%BB%87%E9%A1%B9%E7%9B%AE.md)
- 2. [关注点分离](02-%E5%85%B3%E6%B3%A8%E7%82%B9%E5%88%86%E7%A6%BB.md)
- 3. [管理依赖项](03-%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96%E9%A1%B9.md)
- 4. [API 测试简介](04-API%20%E6%B5%8B%E8%AF%95%E7%AE%80%E4%BB%8B.md)
- 5. [使用 TestClient 进行单元测试](05-%E4%BD%BF%E7%94%A8%20TestClient%20%E8%BF%9B%E8%A1%8C%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95.md)
- 6. [预测端点测试](06-%E9%A2%84%E6%B5%8B%E7%AB%AF%E7%82%B9%E6%B5%8B%E8%AF%95.md)
- 7. [FastAPI 应用中的日志记录](07-FastAPI%20%E5%BA%94%E7%94%A8%E4%B8%AD%E7%9A%84%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95.md)
- 8. [管理配置和敏感信息](08-%E7%AE%A1%E7%90%86%E9%85%8D%E7%BD%AE%E5%92%8C%E6%95%8F%E6%84%9F%E4%BF%A1%E6%81%AF.md)
- 9. [动手实践：重构与测试预测服务](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%87%8D%E6%9E%84%E4%B8%8E%E6%B5%8B%E8%AF%95%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1.md)

章节测验：[在线测验](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi/quiz)
