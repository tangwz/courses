---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "request-response-cycle"
sourceId: 5243
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/request-response-cycle"
title: "FastAPI 请求/响应周期的构成"
description: "理解 FastAPI 应用程序中请求如何处理和响应如何生成。"
order: 7
plots: []
sourceHash: "b71071c9a9c53427de8e685c2b1c07606fc10a3dd4a4f4bfd7df03a9de9d3eea"
sourceCorrections: []
---

运行 FastAPI 应用程序时，使用 Uvicorn 等 ASGI 服务器会启动一个监听传入 HTTP 请求的进程。当一个请求到达并发送回一个响应时，究竟会发生什么？理解这个请求/响应周期为构建高效的 Web API 提供了基础。

让我们分析一下 FastAPI 应用中请求的典型过程：

1. **请求到达：** 客户端（如网页浏览器、移动应用，或使用 `curl` 或 Python `requests` 库的其他服务）向运行 FastAPI 应用程序的服务器地址和端口发送 HTTP 请求（例如，`http://127.0.0.1:8000/items/5?query_param=abc`）。该请求包含：

   - HTTP 方法（例如，`GET`、`POST`、`PUT`、`DELETE`）。
   - 路径（例如，`/items/5`）。
   - 头部（包含元数据，如 `Content-Type`、`Authorization` 等）。
   - 可选的查询参数 (parameter)（例如，`?query_param=abc`）。
   - 可选的请求体（常用于 `POST`、`PUT`，包含数据，通常为 JSON 格式）。
2. **ASGI 服务器处理：** ASGI 服务器（如 Uvicorn）接收原始 HTTP 请求。它解析请求并将其转换为标准化格式，即 ASGI 范围（ASGI scope），这本质上是一个包含所有请求细节的 Python 字典。
3. **FastAPI 接管：** ASGI 服务器将 ASGI 范围（以及用于接收请求体的相关事件消息）传递给 FastAPI 应用程序实例。
4. **路由：** FastAPI 检查请求的路径（`/items/5`）和 HTTP 方法（`GET`）。它将其与您在应用程序中使用 `@app.get("/items/{item_id}")` 等装饰器定义的*路径操作*进行比较。它找到第一个匹配的路径操作函数。
5. **参数解析与验证：** FastAPI 根据您函数的签名自动从请求中解析参数：

   - **路径参数：** 如果匹配的路由包含路径参数（如 `/items/{item_id}` 中的 `{item_id}`），FastAPI 会从路径中提取相应的值（我们示例中的 `5`），并将其转换为您函数签名中声明的类型（例如，`item_id: int`）。
   - **查询参数：** 它提取查询参数（如 `query_param=abc`），并将其与不属于路径部分的函数参数进行匹配。类型提示用于类型转换和基本验证。
   - **请求体：** 对于 `POST` 或 `PUT` 等方法，如果您的函数需要一个请求体（通常使用 Pydantic 模型定义），FastAPI 会读取请求体，解析它（通常假定为 JSON 格式），并根据 Pydantic 模型对其进行验证。我们将在下一章详细讲解 Pydantic。
6. **依赖注入：** FastAPI 的依赖注入系统解决为路径操作函数声明的任何依赖项（更多内容在后续章节中介绍）。
7. **路径操作函数执行：** FastAPI 调用您的路径操作函数（用 `@app.get`、`@app.post` 等装饰的 Python 函数），将提取和验证后的参数作为实参传递。如果函数定义为 `async def`，FastAPI 会等待其执行，如果函数执行 I/O 操作，则允许其他请求并发处理。如果它是常规的 `def` 函数，FastAPI 会在外部线程池中运行它，以避免阻塞主事件循环（这对 CPU 密集型任务（如机器学习 (machine learning)推理 (inference)）很重要）。
8. **处理逻辑：** 您的函数代码执行。您在此处实现端点的具体逻辑，例如从数据库中获取数据、处理输入，或者，在本课程中，加载机器学习模型并执行推理。
9. **响应生成：** 您的函数返回一个结果（例如，字典、列表、Pydantic 模型实例、字符串或 `Response` 对象）。
10. **数据转换：** FastAPI 接收您函数的返回值，并将其转换为适当的 HTTP 响应。

    - 如果您返回字典或列表等 Python 对象，或 Pydantic 模型，FastAPI 会自动将其转换为 JSON 格式，并将 `Content-Type` 头部设置为 `application/json`。
    - 您也可以显式返回 `fastapi.responses.Response`（或其子类，如 `JSONResponse`、`HTMLResponse`），以便更好地控制状态码、头部和响应体。
11. **响应模型验证（可选）：** 如果您在路径操作装饰器中声明了 `response_model`，FastAPI 会根据此模型验证并筛选传出数据，确保响应结构与定义的模式匹配。
12. **响应传输：** FastAPI 将最终的 HTTP 响应（状态码、头部、响应体）发送回 ASGI 服务器。
13. **最终交付：** ASGI 服务器通过网络将 HTTP 响应传输回原始客户端。

这个周期对每个传入的请求都会重复。FastAPI 对类型提示、Pydantic 和异步功能的使用，使参数解析、数据验证和并发处理等步骤高效且对开发者友好。

> 此图展示了请求如何从客户端通过 ASGI 服务器和 FastAPI 到您的应用代码，以及响应返回客户端的过程。

理解这个流程有助于调试问题、有效地组织您的应用逻辑，并理解 FastAPI 组件如何协同工作以提供您的 API，包括为机器学习预测设计的 API。随着课程推进，我们将更详细地讨论每个阶段，特别是使用 Pydantic 进行数据验证以及在路径操作函数中执行机器学习模型。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  提供关于FastAPI功能的全面信息，包括请求/响应流程、路由、依赖注入以及与Pydantic和ASGI的集成。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin, et al. (2024)
  用于理解FastAPI如何利用类型提示处理数据解析、验证以及请求体和响应模型的序列化。
