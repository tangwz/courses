---
course: "fastapi-ml-deployment"
chapter: "structuring-testing-fastapi"
lesson: "organizing-project-routers"
sourceId: 5294
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi/organizing-project-routers"
title: "使用路由组织项目"
description: "使用 APIRouter 将您的应用划分为逻辑模块，便于维护。"
order: 1
plots: []
sourceHash: "a819d7875c3245fbe2ca1a2a0fdfd40e18fc2b21904c52da54fd595b5f604bb8"
sourceCorrections: []
---

随着您的 FastAPI 应用用于部署机器学习 (machine learning)模型而日益发展，将所有 API 端点都放在一个 Python 文件（例如 `main.py`）中很快就会变得难以管理。代码将变得更难查找、维护和调试。API 的不同逻辑部分，例如预测端点、状态检查或模型管理路由，会混杂在一起。

FastAPI 为此提供了一个整洁的解决方案：`APIRouter`。您可以将 `APIRouter` 看作一个迷你 FastAPI 应用。它允许您将相关路径操作（即端点）组合在一起，通常放在独立的 Python 模块中，然后将它们连接回您的主 `FastAPI` 应用实例。这种模块化方法是构建大型、结构良好的服务的基础。

### 介绍 APIRouter

一个 `APIRouter` 的运作方式与主 `FastAPI` 应用对象非常相似。您可以在 `APIRouter` 实例上定义路径操作（使用 `@router.get`、`@router.post` 等装饰器），就像在 `app` 实例上定义一样。

让我们看看如何使用它。假设您想将模型预测端点分离到自己的模块中。

首先，创建一个新文件，例如 `routers/predictions.py`。在此文件中，您导入 `APIRouter` 并创建一个实例：

```python
# routers/predictions.py
from fastapi import APIRouter
from pydantic import BaseModel

# 如果需要，定义您的输入/输出 Pydantic 模型
class PredictionInput(BaseModel):
    feature1: float
    feature2: str
    # ... 其他特征

class PredictionOutput(BaseModel):
    prediction: float
    probability: float | None = None

# 创建路由实例
router = APIRouter()

# 在路由实例上定义路径操作
@router.post("/make", response_model=PredictionOutput)
async def make_prediction(data: PredictionInput):
    """
    接收输入数据并返回模型预测结果。
    （实际的模型加载和推理逻辑将在此处）
    """
    # 模型推理逻辑的占位符
    prediction_value = 1.0 # 替换为实际的 model.predict(...)
    probability_value = 0.85 # 如果您的模型提供概率，则替换

    # 在这里，您通常会：
    # 1. 预处理输入数据 (data.dict())
    # 2. 从您加载的机器学习模型获取预测结果
    # 3. 如果有必要，对结果进行后处理
    # 4. 返回结构化的输出

    return PredictionOutput(prediction=prediction_value, probability=probability_value)

@router.get("/info")
async def get_model_info():
    """
    返回关于预测模型的基本信息。
    """
    return {"model_name": "ExamplePredictor", "version": "1.0"}

# 您可以在此处添加更多与预测相关的端点...
```

请注意，我们使用 `@router.post` 和 `@router.get`，而不是 `@app.post` 和 `@app.get`。函数签名和 Pydantic 模型的使用方式保持完全一致。

### 在主应用中包含路由

现在，您需要让您的主 `FastAPI` 应用了解这个路由。在 `main.py` 中（或者主 `FastAPI` 实例定义的位置），您导入路由对象并使用 `app.include_router()` 方法。

````python
# main.py
from fastapi import FastAPI
from routers import predictions # 导入包含路由的模块

# 创建主 FastAPI 应用实例
app = FastAPI(
    title="ML Model Prediction Service",
    description="API for serving predictions from an ML model.",
    version="0.1.0"
)

# 包含预测路由
app.include_router(predictions.router)

@app.get("/")
async def read_root():
    ```python
    return {"message": "Prediction API!"}
````

# 您也可以在此处包含其他路由

# from routers import monitoring

# app.include\_router(monitoring.router)

````
通过这种设置，在 `routers/predictions.py` 中定义的端点（`/make` 和 `/info`）现在是您主应用的一部分。FastAPI 在内部处理路由。当对 `/make` 或 `/info` 的请求到达时，它将被定向到 `predictions` 路由中对应的函数。

### 使用前缀和标签组织 URL

`include_router` 方法提供有用的参数，有助于更好地组织，特别是在 API 文档和 URL 结构方面。

*   **`prefix`**: 为路由中定义的所有路由添加路径前缀。这非常适合将端点分组到一个共同的路径段下。
*   **`tags`**: 为路由中的所有路由分配标签。这些标签用于在自动生成的 API 文档（如 Swagger UI）中对操作进行分组。

让我们修改 `main.py` 以使用这些参数：

```python
# main.py (updated)
from fastapi import FastAPI
from routers import predictions

app = FastAPI(
    title="ML Model Prediction Service",
    description="API for serving predictions from an ML model.",
    version="0.1.0"
)

# 包含带前缀和标签的预测路由
app.include_router(
    predictions.router,
    prefix="/predict", # predictions.router 中的所有路由现在都以 /predict 开头
    tags=["Predictions"] # 在文档中将这些端点分组到“预测”下
)

@app.get("/")
async def read_root():
    ```python
    return {"message": "Prediction API!"}
````

````
现在，之前在 `/make` 访问的端点将在 `/predict/make` 可用，而 `/info` 将在 `/predict/info` 可用。在您的交互式 API 文档（通常在 `/docs` 找到）中，您将看到这些端点整齐地分组在“预测”标签下。

```graphviz
digraph G {
  rankdir=LR;
  graph [bgcolor=transparent, fontname="Arial"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#495057", fillcolor="#e9ecef", fontcolor="#495057"];
  edge [color="#adb5bd", fontname="Arial", fontsize=9];

  subgraph cluster_main {
    label = "main.py";
    style="filled,rounded";
    color="#dee2e6";
    bgcolor="#f8f9fa";
    main_app [label="app = FastAPI()"];
    include_call [label="app.include_router(\n  predictions.router,\n  prefix='/predict',\n  tags=['Predictions']\n)", shape=invhouse];
    main_app -> include_call [label=" 配置 "];
  }

  subgraph cluster_router {
    label = "routers/predictions.py";
    style="filled,rounded";
    color="#dee2e6";
    bgcolor="#f8f9fa";
    router_instance [label="router = APIRouter()"];
    predict_endpoint [label="@router.post('/make')", shape=ellipse, fillcolor="#a5d8ff"];
    info_endpoint [label="@router.get('/info')", shape=ellipse, fillcolor="#a5d8ff"];
    router_instance -> predict_endpoint;
    router_instance -> info_endpoint;
  }

  include_call -> router_instance [label=" 包含 "];
}
````

> 示意图显示了主应用如何包含预测路由模块。`include_router` 调用指定了 `/predict` 前缀并分配了“预测”标签。

使用 `APIRouter` 是构建可维护和可伸缩的 FastAPI 应用的重要一步。它使您能够：

1. **关注点分离**：将与特定功能（如预测、用户管理、数据预处理）相关的代码保留在独立的模块中。
2. **提高可读性**：文件更小，更易于理解和查找。
3. **促进协作**：不同的团队成员可以同时处理不同的路由，减少合并冲突。
4. **组织 URL**：使用前缀创建逻辑 URL 结构（例如，`/api/v1/predict/...`）。
5. **增强文档**：使用标签对自动生成的 API 文档中的相关端点进行分组。

随着您的机器学习 (machine learning) API 的发展，尽早采用路由将在长期为您节省大量精力，并构成组织良好的项目结构的支柱。

## 参考资料

- [Bigger Applications - Routers](https://fastapi.tiangolo.com/tutorial/bigger-applications/#routers) — Sebastián Ramírez (2024)
  FastAPI官方文档的这一部分是理解和实现`APIRouter`进行模块化应用程序设计的权威来源。
- [Building Data Science Applications with FastAPI](https://www.packtpub.com/product/building-data-science-applications-with-fastapi/9781803240034) — François Voron (2023)
  Publisher: Packt Publishing; Pages: 422
  为构建FastAPI应用程序提供实践指导，包括`APIRouter`的使用，特别适合部署机器学习模型和数据科学解决方案。
- [Building Microservices: Designing Fine-Grained Systems](https://www.oreilly.com/library/view/building-microservices-2nd/9781492034018/) — Sam Newman (2021)
  Publisher: O'Reilly Media; Pages: 612
  虽然不专门针对FastAPI，但本书阐述了基本的软件架构原则，如模块化和关注点分离，`APIRouter`有助于实现这些原则以构建组织良好的API服务。
