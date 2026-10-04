---
course: "fastapi-ml-deployment"
chapter: "data-validation-pydantic"
lesson: "introduction-pydantic"
sourceId: 5249
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic/introduction-pydantic"
title: "Pydantic 简介"
description: "Pydantic 概览及其在 Python 中数据验证和设置管理中的作用。"
order: 1
plots: []
sourceHash: "03a1cf0859ef323f892af6b1e3334c98a55ba938bed4b12c516e212a6959953e"
sourceCorrections: []
---

您的机器学习 (machine learning) API 的可靠性很大程度上取决于其接收数据的质量和结构。向您的机器学习模型发送格式错误或类型不正确的数据可能导致意料之外的错误、不准确的预测，甚至服务中断。手动编写代码来检查传入请求的每个字段、类型和约束既繁琐又容易出错，还会使您的核心应用逻辑变得杂乱。

Pydantic 在此发挥作用。Pydantic 是一个 Python 库，专门用于使用标准 Python 类型注释进行数据验证和设置管理。您无需编写命令式验证代码（例如，`if not isinstance(data['age'], int): raise ValueError(...)`），而是声明性地定义数据应有的*形态*。

Pydantic 利用这些类型提示来解析和验证数据。您可以通过创建继承自 Pydantic 的 `BaseModel` 的类来定义数据结构。此类的属性使用标准 Python 类型（如 `int`、`float`、`str`、`list`、`dict`）以及类型提示来定义。

```python
# 一个说明此思想的简单例子
from pydantic import BaseModel
from typing import List

class InputFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    tags: List[str] = [] # 可选的字符串列表，带默认值
```

在 FastAPI 的环境中，Pydantic 提供了几个主要优点：

- **自动数据验证：** 当您在 FastAPI 路径操作函数中将 Pydantic 模型声明为请求体参数 (parameter)的类型提示时，FastAPI 会自动读取请求体（例如，JSON），解析它，并根据您的模型定义对其进行验证。如果传入数据不符合模型的结构和类型，FastAPI 会自动返回一个标准的 HTTP 422（不可处理实体）错误响应，详细说明验证问题，从而防止无效数据到达您的应用逻辑。
- **数据序列化：** Pydantic 模型可以像解析数据一样简单地序列化数据（将 Python 对象转换为 JSON 等格式）。当您定义 `response_model` 时，FastAPI 会使用此功能，从而确保传出数据也符合特定结构。
- **改进的开发体验：** 使用类型提示使您的代码更具可读性和可维护性。现代 IDE 利用这些提示来提供更好的自动补全、静态分析和重构支持，在运行时之前捕获潜在的类型错误。
- **自动 API 文档：** FastAPI 会自省您的 Pydantic 模型以生成 JSON Schema 定义，这些定义随后用于自动创建交互式 API 文档（通过 Swagger UI 或 ReDoc 等界面）。此文档清晰地向 API 使用者展示预期的请求和响应格式。

以下图表说明了 Pydantic 在 FastAPI 内的请求验证过程中的作用：

> 数据从客户端流向 FastAPI。Pydantic 拦截传入数据，根据定义模型对其进行验证。如果有效，则将 Python 对象传递给应用逻辑；否则，生成错误响应。

虽然基本类型验证功能强大，但 Pydantic 提供了更多功能。您可以定义复杂的嵌套结构，添加约束（如值范围或字符串长度），定义默认值，创建自定义验证逻辑，并管理应用设置。这些能力使其成为构建定义良好且可靠的 API 的重要支持，特别是那些处理机器学习模型结构化数据的 API。

在接下来的章节中，我们将详细阐述如何定义这些 Pydantic 模型，从基本结构开始，逐步转向验证机器学习应用中更常见、更复杂的数据。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin and Pydantic contributors (2024)
  Pydantic的官方文档，提供了关于其使用Python类型提示进行数据验证、设置管理和数据序列化等核心功能的全面信息。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez and FastAPI contributors (2024)
  FastAPI的官方文档，解释了其如何与Pydantic集成，实现请求体自动验证、响应模型序列化和生成OpenAPI文档。
- [FastAPI: From Zero to Production](https://link.springer.com/book/10.1007/978-1-4842-8703-9) — Marcelo Trylesinski (2022)
  Publisher: Apress; DOI: [10.1007/978-1-4842-8703-9](https://doi.org/10.1007/978-1-4842-8703-9)
  一本使用FastAPI构建API的实践指南，广泛介绍了在API开发中Pydantic模型用于定义数据结构、验证和序列化的应用。
