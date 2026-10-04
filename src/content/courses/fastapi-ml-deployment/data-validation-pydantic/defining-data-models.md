---
course: "fastapi-ml-deployment"
chapter: "data-validation-pydantic"
lesson: "defining-data-models"
sourceId: 5250
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic/defining-data-models"
title: "定义数据模型"
description: "学习如何创建 Pydantic 模型来定义数据的结构和类型。"
order: 2
plots: []
sourceHash: "a6a3701894c9ecb681ec54a43d9944f009a363f3ffe6697c122a19f998a5539c"
sourceCorrections: []
---

FastAPI 严重依赖 Pydantic 进行数据验证。这个系统的核心是 **Pydantic 模型**，它们本质上是您定义的 Python 类，用于规定 API 期望接收或发送信息的结构和数据类型。

可以将 Pydantic 模型看作您数据的蓝图。通过定义模型，您声明了数据的“形态”，包括字段名称（例如，JSON 对象中的键）以及每个字段值预期对应的 Python 类型（例如 `int`、`float`、`str`、`bool`，甚至是更复杂的类型，如 `List` 或其他 Pydantic 模型）。

### 使用 `BaseModel` 创建您的第一个模型

为了定义数据模型，您需要创建一个继承自 Pydantic 的 `BaseModel` 的类。在这个类中，您使用标准的 Python 类型注解来声明属性。

让我们看一个简单的例子。假设您有一个机器学习 (machine learning)模型，它根据客户的年龄和之前的交互历史（由布尔值表示）来预测客户是否会点击广告。您可以这样为输入数据定义一个 Pydantic 模型：

```python
from pydantic import BaseModel

class AdPredictionInput(BaseModel):
    age: int
    previous_interaction: bool
    campaign_id: str | None = None # 可选字段，带有默认值
```

在这个 `AdPredictionInput` 模型中：

- 我们继承自 `BaseModel`。
- 我们定义了三个字段：`age`、`previous_interaction` 和 `campaign_id`。
- `age` 声明时带有类型提示 `int`，这意味着 Pydantic（和 FastAPI）将期望此字段的值为整数。
- `previous_interaction` 声明为 `bool`，期望一个布尔值（在 JSON 中为 `true` 或 `false`）。
- `campaign_id` 使用 `str | None = None`。这表示该字段预期为字符串，但也是可选的。如果传入数据中未提供此字段，它将默认为 `None`。

### FastAPI 如何使用这些模型

当您将 `AdPredictionInput` 这样的 Pydantic 模型用作 FastAPI 路径操作函数中参数 (parameter)的类型提示时（我们很快会详细介绍），FastAPI 会自动执行以下操作：

1. **读取请求体：** 它期望传入的请求具有 JSON 请求体。
2. **解析 JSON：** 它将 JSON 数据转换为 Python 对象。
3. **验证数据：** 它检查解析后的数据是否符合您 Pydantic 模型（在我们示例中为 `AdPredictionInput`）中定义的结构和类型。
   - JSON 中是否包含名为 `age` 和 `previous_interaction` 的字段？
   - `age` 的值是否为整数（或可转换为整数）？
   - `previous_interaction` 的值是否为布尔值（或可转换为布尔值）？
   - 如果 `campaign_id` 存在，它是否为字符串？
4. **提供数据：** 如果验证成功，FastAPI 会将经过验证的数据作为您的 Pydantic 模型实例传递给您的函数。
5. **生成错误：** 如果验证失败（例如，缺少必需字段、数据类型不正确），FastAPI 会自动生成详细的 JSON 错误响应，准确指出问题所在，而无需您编写任何特定的验证错误处理代码。

这种使用 `BaseModel` 和类型提示的声明式方法显著简化了数据验证。您只需定义一次预期结构，FastAPI 便会处理强制执行，从而使您的 API 端点更整洁，更专注于核心逻辑，例如与您的机器学习 (machine learning)模型交互。在接下来的部分中，我们将了解如何将这些模型应用于请求体、响应和其他参数类型。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/) — Pydantic Development Team (2024)
  Pydantic 的官方文档，提供数据模型、验证和使用方法的全面信息。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI 的官方文档，详细说明了 Pydantic 模型如何集成用于请求体解析和数据验证。
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/) — Guido van Rossum, Jukka Lehtosalo, Łukasz Langa (2015)
  为 Python 引入类型提示的奠基性文档，阐述了其语法和基本原理。
- [Building Data Science Applications with FastAPI](https://www.manning.com/books/building-data-science-applications-with-fastapi) — Cheuk Ting Ho, Brian Okken (2023)
  Publisher: Manning Publications
  一本使用 FastAPI 开发数据科学应用程序的实用指南，包含 Pydantic 模型用于数据处理的示例。
