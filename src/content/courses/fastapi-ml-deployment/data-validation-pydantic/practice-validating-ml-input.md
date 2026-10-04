---
course: "fastapi-ml-deployment"
chapter: "data-validation-pydantic"
lesson: "practice-validating-ml-input"
sourceId: 5268
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic/practice-validating-ml-input"
title: "动手实践：验证机器学习输入数据"
description: "动手实践创建 Pydantic 模型，以验证示例机器学习模型的输入数据。"
order: 8
plots: []
sourceHash: "e98204a55a709f5e37eeacbd6cb8c7a919660538b4d2d9158bb52a49fe92c45f"
sourceCorrections: []
---

将建立一个 Pydantic 模型，用于验证机器学习 (machine learning)模型的输入数据，特别是那些依据特定属性来预测房价的模型。在将输入数据送入任何模型进行预测之前，验证其符合预期的结构和限制非常重要。

设想我们的房价预测模型需要以下输入属性：

- `area_sqft`：居住面积，单位平方英尺（必须是正数）。
- `bedrooms`：卧室数量（必须是非负整数）。
- `bathrooms`：浴室数量（可以包含半浴室，所以可以是正浮点数，例如 1.5）。
- `region`：房屋所在地区域（必须是“North”、“South”、“East”、“West”之一）。

我们可以定义一个 Pydantic 模型来表示并验证这种结构。

### 定义输入架构

首先，我们为允许的区域定义一个 `Enum`，然后创建我们的 Pydantic `BaseModel`。

```python
# models.py
from pydantic import BaseModel, Field, validator
from enum import Enum

class HouseRegion(str, Enum):
    """房屋位置允许的区域。"""
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"

class HouseFeatures(BaseModel):
    """房价预测模型的输入属性。"""
    area_sqft: float = Field(..., gt=0, description="居住面积（平方英尺），必须为正数。")
    bedrooms: int = Field(..., ge=0, description="卧室数量，必须为非负数。")
    bathrooms: float = Field(..., gt=0, description="浴室数量，必须为正数（例如，1.5 表示 1 个完整浴室，1 个半浴室）。")
    region: HouseRegion = Field(..., description="房屋所在的区域。")

    # 如果需要自定义验证器的一个例子，尽管Field约束已涵盖这种情况
    # @validator('area_sqft')
    # def area_must_be_positive(cls, v):
    #     if v <= 0:
    #         raise ValueError('Area must be positive')
    #     return v

    class Config:
        # 为文档提供示例数据
        schema_extra = {
            "example": {
                "area_sqft": 1500.5,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "region": "North"
            }
        }
```

在这个 `HouseFeatures` 模型中：

- 我们使用类型提示（`float`、`int`、`HouseRegion`）进行基本类型验证。
- `Field` 从 Pydantic 导入，用于添加限制：
  - `...` 表示该字段为必填。
  - `gt=0` 表示“大于 0”。
  - `ge=0` 表示“大于或等于 0”。
  - `description` 有助于在 API 架构中描述字段。
- 我们使用 `HouseRegion` 枚举来将 `region` 字段限制为特定的字符串值。
- `Config.schema_extra` 提供了一个示例载荷，它将出现在自动生成的 API 文档中。
- 示例中展示了一个被注释掉的自定义 `@validator`，尽管 Pydantic 内置的 `Field` 约束通常足以满足像正数这样的简单检查。

### 与 FastAPI 端点集成

现在，我们创建一个简单的 FastAPI 应用程序，它使用此模型来验证预测端点的传入请求数据。

```python
# main.py
from fastapi import FastAPI, HTTPException
from models import HouseFeatures # 假设 models.py 在同一目录下

app = FastAPI(
    title="房价预测 API",
    description="根据房屋属性预测房价的 API。",
    version="0.1.0",
)

@app.post("/predict/house_price")
async def predict_house_price(features: HouseFeatures):
    """
    根据房屋属性预测价格。

    此端点接受房屋属性并返回一个占位预测。
    使用 `HouseFeatures` 模型执行输入验证。
    """
    # 在实际应用中，您会在此处加载您的机器学习模型
    # 并使用已验证的属性：features.area_sqft, features.bedrooms, 等。
    # 例子：prediction = model.predict([[features.area_sqft, features.bedrooms, ...]])

    # 对于本次练习，我们只返回已验证的数据和一个虚拟预测
    print(f"收到有效属性：{features.dict()}")

    # 虚拟预测逻辑
    estimated_price = (features.area_sqft * 100) + (features.bedrooms * 5000) + (features.bathrooms * 3000)
    if features.region == "North":
        estimated_price *= 1.2
    elif features.region == "West":
        estimated_price *= 1.1

    return {"validated_features": features, "estimated_price": round(estimated_price, 2)}

# 运行此应用程序：uvicorn main:app --reload
```

在这里，`/predict/house_price` 端点需要一个 POST 请求。通过使用我们的 `HouseFeatures` 模型对 `features` 参数 (parameter)进行类型提示（`features: HouseFeatures`），FastAPI 会自动执行以下操作：

1. 读取传入请求的 JSON 正文。
2. 尝试根据 `HouseFeatures` 模型解析和验证数据。
3. 如果验证成功，函数中的 `features` 变量将是 `HouseFeatures` 的一个实例，其中包含已验证的数据。
4. 如果验证失败（例如，字段缺失、类型不正确、值超出限制），FastAPI 会自动返回一个 `422 Unprocessable Entity` HTTP 错误响应，详细说明验证错误。

### 测试验证

使用 Uvicorn 运行应用程序：
`uvicorn main:app --reload`

现在，您可以使用 `curl` 等工具或 FastAPI 的交互式文档（可在 `http://127.0.0.1:8000/docs` 访问）测试该端点。

**有效请求：**

```bash
curl -X POST "http://127.0.0.1:8000/predict/house_price" \
-H "Content-Type: application/json" \
-d '{
    "area_sqft": 2150.75,
    "bedrooms": 4,
    "bathrooms": 3,
    "region": "West"
}'
```

*预期响应（状态码 200）：*

```json
{
  "validated_features": {
    "area_sqft": 2150.75,
    "bedrooms": 4,
    "bathrooms": 3,
    "region": "West"
  },
  "estimated_price": 268582.5
}
```

**无效请求（面积为负）：**

```bash
curl -X POST "http://127.0.0.1:8000/predict/house_price" \
-H "Content-Type: application/json" \
-d '{
    "area_sqft": -100,
    "bedrooms": 2,
    "bathrooms": 1,
    "region": "South"
}'
```

*预期响应（状态码 422）：*

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "area_sqft"
      ],
      "msg": "确保此值大于 0",
      "type": "value_error.number.not_gt",
      "ctx": {
        "limit_value": 0
      }
    }
  ]
}
```

**无效请求（区域不正确）：**

```bash
curl -X POST "http://127.0.0.1:8000/predict/house_price" \
-H "Content-Type: application/json" \
-d '{
    "area_sqft": 1200,
    "bedrooms": 2,
    "bathrooms": 1.5,
    "region": "Central"
}'
```

*预期响应（状态码 422）：*

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "region"
      ],
      "msg": "值不是有效的枚举成员；允许的值为：“North”、“South”、“East”、“West”",
      "type": "type_error.enum",
      "ctx": {
        "enum_values": [
          "North",
          "South",
          "East",
          "West"
        ]
      }
    }
  ]
}
```

这个动手练习说明了 Pydantic 模型与 FastAPI 端点结合时，如何提供一种强大且声明式的方法来强制执行数据约定。这种自动验证可以保护您的下游逻辑（包括机器学习 (machine learning)模型推理 (inference)）免受处理格式错误或无意义的输入数据的影响，从而使应用程序更稳定且易于维护。清晰的错误消息也极大地帮助 API 调用者调试他们的请求。

> 在 FastAPI 端点中使用 Pydantic 进行请求验证的流程。有效数据会进入端点逻辑，而无效数据会触发自动错误响应。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin and Pydantic Contributors (2024)
  Pydantic 数据验证功能的官方资源，涵盖 `BaseModel`、`Field` 约束和自定义验证器。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (Tiangolo) and FastAPI Contributors (2024)
  在 FastAPI 应用程序中集成 Pydantic 进行请求体验证、自动 API 文档和错误处理的权威指南。
- [Building APIs with FastAPI: Develop, secure, and deploy robust and high-performance RESTful APIs](https://www.packtpub.com/product/building-python-web-apis-with-fastapi/9781800569731) — Vitor J. Nexus (2021)
  Publisher: Packt Publishing
  一本实用书籍，为使用 FastAPI 构建 API 提供结构化指导，包括使用 Pydantic 进行数据建模和验证的最佳实践。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书讨论了构建生产就绪的机器学习系统，强调了数据验证和质量对系统可靠性的重要性。
