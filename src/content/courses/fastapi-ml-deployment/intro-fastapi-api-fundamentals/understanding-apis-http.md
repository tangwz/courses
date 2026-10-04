---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "understanding-apis-http"
sourceId: 5232
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/understanding-apis-http"
title: "理解 API 和 HTTP 方法"
description: "回顾 Web API 的核心原理、REST 原则以及标准 HTTP 方法（GET、POST、PUT、DELETE）。"
order: 3
plots: []
sourceHash: "7719323f10d62a15c6228ce21946d3d6dd0b6e5c333ed8a6b91575dec19fc4cd"
sourceCorrections: []
---

API 和 HTTP 方法是实现网络交互的基础机制。掌握这些基本概念对于开发包括 FastAPI 应用在内的网络应用程序很有帮助。

**API**，即应用程序编程接口，充当中间人，允许两个不同的软件应用相互通信。可以将其想象成餐厅的菜单。菜单（API）提供一份您可以点的菜品（操作）列表，以及每道菜的描述（数据格式）。您（客户端应用）根据菜单发出请求，厨房（服务器应用）准备好菜品并将其送回给您。您不需要知道确切的食谱或厨房如何运作；您只需知道如何从菜单点餐。

在网络开发和模型部署中，我们通常使用 **Web API**。这些 API 使用超文本传输协议（HTTP）以便 **客户端**（如网页浏览器、移动应用或其他后端服务）与 **服务器**（我们的 FastAPI 应用和机器学习 (machine learning)模型所在之处）之间进行通信。客户端向服务器发送 HTTP 请求，服务器发送回 HTTP 响应。

### 客户端-服务器模型

这种交互遵循标准的客户端-服务器模型：

1. **客户端请求：** 客户端通过向特定的服务器地址（URL）发送请求来发起通信。此请求包含客户端希望服务器执行什么操作的详情以及可能的一些数据。
2. **服务器处理：** 服务器接收请求，进行处理（例如，获取数据、执行计算、进行模型推理 (inference)），并确定适当的结果。
3. **服务器响应：** 服务器将响应发送回客户端，其中包含请求的信息或操作结果，以及指示请求是否成功的状态码。

> 使用 HTTP 进行客户端-服务器交互的简化视图。

### REST：Web API 的一种风格

许多 Web API，包括使用 FastAPI 构建的，遵循 **REST**（表述性状态转移）的原则。REST 不是一个严格的协议，而是一种架构风格，它定义了一系列约束，用于构建可伸缩、无状态且易于维护的 Web 服务。主要思想包括：

- **资源：** 信息以资源的形式表示（例如，特定用户、产品、预测结果）。每个资源通过唯一的 URL（统一资源定位符）识别，如 `/models/iris-classifier` 或 `/predict`。
- **表述：** 客户端与这些资源的表述进行交互，通常采用 JSON（JavaScript 对象表示法）或 XML 等格式。当您请求一个资源时，服务器会返回其当前状态的表述。
- **无状态性：** 客户端向服务器发出的每个请求都必须包含理解和完成该请求所需的所有信息。服务器在请求之间不存储任何客户端上下文 (context)。这提高了可靠性和可伸缩性。
- **标准方法：** REST 使用标准的 HTTP 方法对资源执行操作。

### 常用 HTTP 方法

HTTP 定义了几种请求方法（常称为“动词”），它们表示希望对由 URL 标识的资源执行的动作。FastAPI 使用这些方法将传入的请求路由到您代码中正确的 Python 函数。您会遇到最常见的方法是：

- **GET：** 用于获取资源的表述。这是一种只读操作，应是*安全的*（多次调用产生相同结果）和*幂等的*（调用一次与调用多次的效果相同）。
  - *机器学习 (machine learning)示例：* 请求已部署模型的元数据（`GET /models/info/resnet50`）。获取过去的预测结果（`GET /predictions/123`）。
- **POST：** 用于向指定资源提交数据，通常会导致服务器上的状态更改或附带效应。它常用于创建新资源或触发处理任务。POST 请求既不安全，也未必幂等。
  - *机器学习示例：* 将输入特征（例如，图像数据、文本）发送到预测端点（`POST /predict/image`）。提交数据来训练或微调 (fine-tuning)模型（尽管通常离线完成）。
- **PUT：** 用于用请求负载替换目标资源的当前表述。如果资源存在，则更新；如果不存在，则可能创建。PUT 被认为是幂等的。
  - *机器学习示例：* 更新已部署模型的配置（`PUT /models/config/iris-classifier`）。完全替换模型文件（通过直接 API 较不常见，但可行）。
- **DELETE：** 用于删除指定资源。DELETE 是幂等的。
  - *机器学习示例：* 删除特定的已部署模型版本（`DELETE /models/version/spam-filter-v1`）。删除存储的预测结果（`DELETE /predictions/456`）。

其他方法如 `PATCH`（用于部分更新）、`HEAD`（类似于 GET 但没有响应体）和 `OPTIONS`（获取资源的通信选项）也存在，但与 GET 和 POST 相比，它们在基本的机器学习部署 API 中使用频率较低。

### 请求和响应结构

一个 **HTTP 请求** 通常包含：

1. **方法：** HTTP 动词（GET、POST、PUT、DELETE 等）。
2. **URL：** 服务器上资源的地址。
3. **头部：** 关于请求的元数据（例如，`Content-Type: application/json` 表示请求体的格式，`Authorization` 用于凭证）。
4. **请求体（可选）：** 随请求发送的数据负载，常与 POST 和 PUT 方法一起使用（例如，机器学习 (machine learning)模型的输入特征，以 JSON 格式）。

一个 **HTTP 响应** 通常包含：

1. **状态码：** 一个三位数字代码，指示结果（例如，`200 OK` 成功，`201 Created` 已创建，`400 Bad Request` 错误请求，`404 Not Found` 未找到，`500 Internal Server Error` 内部服务器错误）。
2. **头部：** 关于响应的元数据（例如，`Content-Type: application/json`，`Content-Length`）。
3. **响应体（可选）：** 发送回客户端的数据负载（例如，机器学习模型的预测结果）。

FastAPI 提供便捷的方式来定义端点，这些端点与特定的 URL 路径和 HTTP 方法对应。它自动处理请求数据（如 JSON 请求体）的解析和响应数据的格式化，基于您定义的 Python 类型和 Pydantic 模型，这将在下一章中介绍。了解这些基本的 HTTP 原理对于为您的机器学习模型构建高效的 Web API 很有帮助。

## 参考资料

- [Architectural Styles and the Design of Network-based Software Architectures](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm) — Roy Thomas Fielding (2000)
  Journal: Doctoral Dissertation; Publisher: University of California, Irvine
  引入并定义了表征状态传输（REST）架构风格的开创性博士论文。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez and FastAPI Contributors (2023)
  FastAPI的官方文档，提供了如何使用Python和Pydantic高效构建API的详细信息。
