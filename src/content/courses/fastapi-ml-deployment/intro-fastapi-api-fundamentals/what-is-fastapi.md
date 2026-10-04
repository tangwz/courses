---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "what-is-fastapi"
sourceId: 5227
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/what-is-fastapi"
title: "什么是FastAPI？"
description: "了解FastAPI Web框架的特点和用途。"
order: 1
plots: []
sourceHash: "22e4868f0c8495eda8239cceefaf671eadaf7486836aa3f9bb95555411bbf33d"
sourceCorrections: []
---

FastAPI是一个现代Python Web框架，专门设计用于创建API，特别是RESTful API。它因其高性能和开发者友好的特性而表现突出，是服务机器学习 (machine learning)模型等应用的优秀选择。

FastAPI建立在两个主要组成部分之上：Starlette和Pydantic，它继承并结合了它们的优点：

1. **Starlette：** 提供底层的ASGI（异步服务器网关接口）支撑。ASGI是Flask和Django等框架使用的传统WSGI（Web服务器网关接口）的后续。ASGI对异步操作的原生支持使FastAPI应用能够高效处理许多并发连接，这对于Web应用中常见的I/O密集型任务（如网络请求或文件读取）尤其有益。
2. **Pydantic：** 处理数据验证、序列化和文档生成。FastAPI广泛使用Python类型提示。通过基于这些类型提示使用Pydantic模型定义数据结构，FastAPI自动验证传入的请求数据，序列化传出的响应数据，并生成交互式API文档。

### 核心特点

FastAPI在设计时考虑了多个目标，直接解决了旧框架中存在的一些局限性：

- **高性能：** 借助Starlette和异步编程（`async`/`await`），FastAPI是现有最快的Python Web框架之一，对于I/O密集型任务，其性能常能媲美NodeJS和Go。在服务机器学习 (machine learning)模型时，这种速度很有利，因为最小化API延迟通常很重要。

  > 请求处理模型的简化比较。像FastAPI这样的异步框架可以在等待I/O操作时处理其他任务，从而提高并发性。
- **编码快速：** 旨在提高开发速度。自动数据验证和文档生成等功能大大减少了样板代码。
- **更少错误：** 使用Python类型提示可以实现出色的编辑器支持（自动补全、类型检查），在开发阶段而非运行时捕获许多错误。Pydantic的严格验证可以阻止无效数据在您的应用程序中传播。
- **直观易用：** 出色的编辑器支持和清晰的结构使开发更轻松。自动文档意味着您的API规范始终与代码保持同步。
- **易于使用：** 设计为易于使用和学习。文档全面且提供许多示例。
- **基于标准：** 完全符合API的开放标准：用于API文档的OpenAPI（以前称为Swagger）和用于数据验证的JSON Schema。

### 为什么将FastAPI用于机器学习 (machine learning)部署？

虽然本章节提到了API的一般基础知识，但FastAPI的特定功能与部署机器学习模型的要求很好地契合：

- **数据验证：** 机器学习模型通常要求输入数据采用非常特定的格式（例如，具有特定数据类型的某些特征）。Pydantic模型允许您精确定义这种预期的输入结构。FastAPI会自动根据此模型验证传入请求，如果数据不正确，则会在数据到达模型预测代码之前返回清晰的错误。类似地，您也可以定义预测输出的结构。
- **性能：** 尽管机器学习模型推理 (inference)本身可能是CPU密集型（稍后讨论），但周围的API操作（接收请求、预处理输入、后处理输出、网络通信）能从ASGI的异步处理中极大受益，使服务器能够高效管理多个并发预测请求。
- **自动文档：** 自动生成的交互式文档（通常通过Swagger UI在`/docs`和`/redoc`提供）非常有帮助。它让数据科学家、前端开发人员或您的机器学习API的其他使用者能够准确理解如何构建他们的请求以及预期何种响应，这些信息可以直接从运行中的应用程序获取。他们甚至可以直接在浏览器中测试端点。
- **Python生态系统：** 作为一个Python框架，FastAPI自然地与数据科学和机器学习的庞大Python生态系统（scikit-learn、TensorFlow、PyTorch、Pandas、NumPy等）集成。

本质上，FastAPI提供了一种现代化、高性能且开发者友好的方式，将您训练好的机器学习模型封装成可靠的Web API，弥合了模型开发与生产部署之间的差距。接下来的部分将介绍API的原理并设置您的环境，以便开始使用FastAPI进行构建。

## 参考资料

- [FastAPI](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2023)
  提供FastAPI框架的全面文档、教程和说明，包括其设计原则以及与Starlette和Pydantic的集成。
- [Pydantic](https://docs.pydantic.dev/) — Samuel Colvin (2023)
  Pydantic的官方文档，这是一个使用Python类型注解的数据验证和设置管理库。
