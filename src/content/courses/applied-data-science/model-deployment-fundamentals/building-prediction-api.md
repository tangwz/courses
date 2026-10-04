---
course: "applied-data-science"
chapter: "model-deployment-fundamentals"
lesson: "building-prediction-api"
sourceId: 1907
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals/building-prediction-api"
title: "构建模型预测的REST API"
description: "使用Flask或FastAPI创建简单的REST API来提供模型预测服务。"
order: 3
plots: []
sourceHash: "40cd8adf8ef5cbd7132e18887756de98d3d81445a8d7a801b1cb58ece7fd59e2"
sourceCorrections: []
---

为了使训练并保存的机器学习 (machine learning)模型发挥作用，需要创建一个接口，以便其他应用程序或用户可以向其发送数据并接收预测。实现此目的的常用且有效方法是将模型封装到Web应用程序编程接口（API）中，特别是REST（表征性状态传输）API。

REST API使用标准HTTP方法（如GET、POST、PUT、DELETE）来允许客户端（请求预测）和服务器（托管模型并计算预测）之间进行通信。对于模型预测，客户端通常通过HTTP POST请求将新的输入数据发送到服务器上一个特定的URL端点。服务器使用加载的模型处理此数据，并在HTTP响应中将预测返回，通常格式为JSON。

此方法有几个优点：

- **解耦：** 客户端应用程序无需了解模型的实现细节，甚至不必用相同的编程语言编写（例如，Java应用程序可以从Python模型API请求预测）。
- **可伸缩性：** Web API可以使用标准Web基础设施独立伸缩。
- **可访问性：** 模型通过网络变得可访问，支持诸如Web应用程序、移动应用或批处理预测管道之类的应用场景。

### 选择Web框架

虽然你可以使用Python内置的`http.server`从头开始构建Web服务器，但使用Web框架会显著简化开发。框架处理HTTP解析、请求路由和响应生成的复杂性。对于Python，构建模型API的两个流行选择是：

1. **Flask：** 一个轻量级“微框架”，相对容易上手。它提供基本功能并通过扩展提供灵活性。
2. **FastAPI：** 一个现代、高性能的框架，基于Starlette和Pydantic构建。它使用Python类型提示提供自动数据验证、交互式API文档（Swagger UI和ReDoc）、异步请求处理，并且通常比Flask有更好的性能。

对于构建模型服务API，FastAPI由于其内置的数据验证和自动文档功能而常被优先选择，这有助于确保正确使用并简化集成。我们的示例将使用FastAPI。

### 设置环境

在构建API之前，请确保已安装必要的库。你通常需要框架本身、一个ASGI服务器（如Uvicorn）来运行FastAPI、用于你的模型的库（例如`scikit-learn`），以及用于保存/加载的库（例如`joblib`）。

```bash
pip install fastapi uvicorn scikit-learn joblib pydantic pandas
```

### 加载模型

在你的API应用程序代码中，首先需要加载你在上一步保存的序列化模型。在API服务器启动时只加载模型*一次*，而不是为每个传入的预测请求重新加载，这很重要。加载模型可能很耗时，重复加载会引入显著延迟。

```python
# main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

# 定义保存的模型文件路径
MODEL_DIR = os.environ.get("MODEL_DIR", ".") # 使用环境变量或当前目录
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

# 在应用程序启动时加载模型
try:
    model = joblib.load(MODEL_PATH)
    print(f"模型已成功从 {MODEL_PATH} 加载")
except FileNotFoundError:
    print(f"错误：模型文件未在 {MODEL_PATH} 找到")
    # 适当处理错误 - 可能退出或使用默认/空模型
    model = None # 或者抛出异常
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None

# 初始化FastAPI应用
app = FastAPI(title="模型预测API", version="1.0")
```

> 此代码尝试加载名为`model.joblib`的模型，加载路径由`MODEL_DIR`环境变量或当前目录指定。其中包含错误处理，以应对文件不存在或其他加载错误的情况。

### 定义输入和输出数据模式

FastAPI的一个重要优势是它使用Pydantic进行数据验证。通过定义Pydantic模型（继承自`BaseModel`的类），你可以指定传入请求和传出响应的预期结构和数据类型。FastAPI使用这些模型自动解析和验证JSON请求体，并将Python对象序列化回JSON响应。

假设我们的模型基于四个数值特征预测类别标签。我们可以这样定义模式：

```python
# main.py (续)

class InputFeatures(BaseModel):
    """定义输入数据的结构"""
    feature1: float
    feature2: float
    feature3: float
    feature4: float

    # 例如，如果需要可以强制执行约束
    # @validator('feature1')
    # def feature1_must_be_positive(cls, v):
    #     if v <= 0:
    #         raise ValueError('feature1必须是正数')
    #     return v

class PredictionOut(BaseModel):
    """定义预测输出的结构"""
    predicted_class: int # 对于回归可以是浮点数，对于类别名称可以是字符串
    probability: float | None = None # 可选：包含预测概率
```

> `InputFeatures`期望一个包含四个键（`feature1`到`feature4`）的JSON对象，每个键都有一个浮点值。`PredictionOut`定义了JSON响应的结构，其中包含预测的类别。

### 创建预测端点

现在，我们定义接收数据并返回预测的API端点。我们使用FastAPI的装饰器语法（`@app.post(...)`）将URL路径（例如`/predict`）和HTTP方法（POST）与Python函数关联起来。

```python
# main.py (续)

@app.get("/")
async def read_root():
    """提供基本API信息的根端点。"""
    return {"message": "请使用 /predict 端点。"}

@app.post("/predict", response_model=PredictionOut)
async def predict(features: InputFeatures):
    """
    通过POST请求接收输入特征，进行预测，
    并返回预测类别。
    """
    if model is None:
        # 处理模型加载失败的情况
        raise HTTPException(status_code=503, detail="模型不可用")

    # 将输入数据转换为模型期望的格式
    # 例如：scikit-learn模型通常期望2D数组状结构
    # 列的顺序必须与训练时使用的顺序一致！
    input_df = pd.DataFrame([features.model_dump()]) # Pydantic v2 使用 model_dump()

    # 如果你的模型对列顺序敏感，请确保列顺序
    # feature_order = ['feature1', 'feature2', 'feature3', 'feature4'] # 定义期望的顺序
    # input_df = input_df[feature_order]

    try:
        # 进行预测
        prediction_result = model.predict(input_df)
        predicted_class = int(prediction_result[0]) # 假设predict返回一个数组

        # 可选：如果模型支持（例如，逻辑回归、树模型），获取预测概率
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            # 假设是二分类，获取预测类别的概率
            prediction_proba = float(probabilities[0, predicted_class])

        # 根据PredictionOut模式返回格式化的预测结果
        return PredictionOut(predicted_class=predicted_class, probability=prediction_proba)

    except Exception as e:
        # 处理预测过程中可能出现的错误
        # 记录错误以进行调试
        print(f"预测时出错: {e}")
        raise HTTPException(status_code=500, detail="预测失败")
```

> 此代码定义了一个接受POST请求的`/predict`端点。函数`predict`接受一个`features`参数 (parameter)，其类型已用我们的`InputFeatures` Pydantic模型进行提示。FastAPI自动根据此模式验证传入的JSON体。函数内部，输入数据被转换为Pandas DataFrame（scikit-learn模型的常用格式），调用模型的`predict`方法，结果会根据`PredictionOut`模式返回，并进行格式化。其中包含模型加载和预测执行的基本错误处理。

### 运行API服务器

要运行此FastAPI应用程序，你可以在终端中使用ASGI服务器（如Uvicorn）：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `main`：指代Python文件`main.py`。
- `app`：指代`main.py`中创建的FastAPI实例（例如，`app = FastAPI()`）。
- `--reload`：代码更改时启用自动重载（对开发有用）。
- `--host 0.0.0.0`：使服务器可在网络上的其他机器访问（仅用于本地访问请使用`127.0.0.1`）。
- `--port 8000`：指定服务器监听的端口号。

服务器运行后，你可以通过将Web浏览器导航到`http://localhost:8000/docs`（Swagger UI）或`http://localhost:8000/redoc`来访问自动生成的交互式文档。

### 测试API

你可以使用各种工具测试正在运行的API端点：

- **FastAPI Docs：** 交互式`/docs`界面允许你直接从浏览器发送测试请求。
- **curl：** 一个用于发出HTTP请求的命令行工具。

  ```bash
  curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
      "feature1": 5.1,
      "feature2": 3.5,
      "feature3": 1.4,
      "feature4": 0.2
  }'
  ```
- **Python `requests` 库：**

  ```python
  import requests
  import json

  api_url = "http://localhost:8000/predict"
  input_data = {
      "feature1": 5.1,
      "feature2": 3.5,
      "feature3": 1.4,
      "feature4": 0.2
  }

  response = requests.post(api_url, json=input_data)

  if response.status_code == 200:
      prediction = response.json()
      print(f"预测响应: {prediction}")
  else:
      print(f"错误: {response.status_code}")
      print(response.text)
  ```

### API请求-响应流程图

以下图表描绘了当客户端从模型API请求预测时的典型流程：

> 客户端将包含输入特征的POST请求发送到API服务器。服务器验证输入、进行格式化，使用已加载的模型进行预测，格式化输出，并将预测结果在响应中发送回客户端。

构建REST API是将机器学习 (machine learning)模型投入实际应用的一个基础步骤。FastAPI等框架简化了此过程，使你能够创建文档良好且性能优秀的预测服务。此API层充当了你训练好的模型与需要其预测能力的应用程序之间的桥梁。下一步通常是将此API服务打包以便更容易部署，这引向了容器化。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  使用 FastAPI 构建 Web API 的基本指南，涵盖核心概念、Pydantic 和部署。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin (2024)
  Pydantic 的官方文档，提供数据验证、设置管理和序列化的详细信息。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH39daDovMPEgRik4Cmc09OI3Cb-MgXGjnZrQgEArZ9Ygx-VT01RLdPHNsMc7p0EthP_CsXo4eMVBnqZ0qmOxrg85UUbNYVorQGpDzdaSAr8yizxqKdq3NOPFIEop3EJxzskeT9aFdO21hKtEv50ekuA17dfOCWe3RNVm55heQHCTGP06OFazggZISIitq_NZkP2m2wGtfkXaUd2Ac7vtO3aDKwK0wdAB5WetQhMfuX0d4Gh0AYPSbY5sCeWnnQPA-2TOU01P-J) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 250
  涵盖将机器学习模型部署到生产环境的实际方面，包括用于服务预测的 API 设计。
- [API Design Patterns](https://www.manning.com/books/api-design-patterns) — JJ Geewax (2021)
  Publisher: Manning Publications
  提供设计稳健且可扩展 API 的实用模式和最佳实践，包括 REST 原则。
