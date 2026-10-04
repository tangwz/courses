---
course: "fastapi-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi"
sourceId: 979
chapter: "structuring-testing-fastapi"
title: "组织和测试 FastAPI 应用"
order: 4
description: "使用路由器组织 FastAPI 项目，管理依赖，并为机器学习 API 实施测试策略。"
hasQuiz: true
---

在学习了如何创建端点、使用 Pydantic 处理数据以及集成机器学习模型之后，我们现在将注意力转向构建更易于维护和可靠的应用程序。随着您的机器学习 API 变得越来越复杂，适当的组织和测试对于项目的持续性变得重要。

本章提供关于如何有效地组织 FastAPI 项目和实施测试策略的指导。你将学习如何使用 `APIRouter` 来模块化代码，应用关注点分离的原则，管理项目依赖，以及使用 FastAPI 的 `TestClient` 编写单元测试和集成测试。我们还将介绍日志记录实践以及处理应用程序配置和秘密的方法。目标是让你掌握开发可扩展和可靠的机器学习部署服务所需的技术。
