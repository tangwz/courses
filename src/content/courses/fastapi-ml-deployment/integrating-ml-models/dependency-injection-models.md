---
course: "fastapi-ml-deployment"
chapter: "integrating-ml-models"
lesson: "dependency-injection-models"
sourceId: 5288
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/dependency-injection-models"
title: "模型加载的依赖注入"
description: "使用FastAPI的依赖注入系统管理请求处理器中的模型加载和访问。"
order: 7
plots: []
sourceHash: "1d054da86e460e61e7cf701af23343de15a2503ca2a0383ca3fe047d6b93ccac"
sourceCorrections: []
---

机器学习 (machine learning)模型被序列化并加载到应用环境中。一个主要考量是如何让这些已加载模型在API端点函数中可用以进行预测。

你可以将模型加载到一个全局变量中，并直接在端点函数中访问它。这种方法虽然简单，但随着应用规模增大，可能会使测试和维护变得困难。全局状态可能导致代码不同部分之间紧密关联，并且难以替换实现，例如在测试时使用模拟模型。

FastAPI通过其**依赖注入**系统提供了一种强大而优雅的方案。依赖注入是一种设计模式，对象所需的组件（其依赖项）是从外部源“注入”到它内部的，而不是由对象自身创建。在FastAPI中，这个系统允许你为路径操作函数（处理请求的函数，如`@app.post("/predict")`）声明依赖项，FastAPI会在调用端点时负责提供这些依赖项。

### FastAPI中依赖注入的运作方式

FastAPI的依赖注入主要使用`Depends`函数。你在路径操作函数的参数 (parameter)中使用`Depends`声明一个依赖项，并传入一个可调用对象（如另一个函数）。当该端点接收到请求时，FastAPI会：

1. 调用依赖函数（传入`Depends`的可调用对象）。
2. 获取该函数的返回值。
3. 将该返回值作为参数传递给你的路径操作函数。

这种机制允许你将*提供*资源（如机器学习 (machine learning)模型）的逻辑与在端点中*使用*该资源的逻辑分开。

### 使用`Depends`访问模型

我们来看看这如何应用于访问机器学习 (machine learning)模型。假设你已经在应用启动时将模型加载到内存中，也许将其存储在一个可在你的应用模块中访问的变量中（我们称之为`loaded_model`）。

首先，定义一个简单的依赖函数，其唯一职责是返回已加载的模型：

```python
# 假设 'model_loader.py' 在启动时处理模型加载
from .model_loader import loaded_model

# 依赖函数
def get_model():
  """返回预加载的机器学习模型。"""
  return loaded_model
```

现在，在需要执行预测的API端点函数中，你可以声明它依赖于`get_model`的结果：

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
# 假设 'get_model' 如上所示定义
from .dependencies import get_model
# 假设 'ml_model_type' 是你加载模型的类型提示
from typing import Any # 如果可能，用你的实际模型类型替换 Any

app = FastAPI()

class PredictionInput(BaseModel):
    feature1: float
    feature2: int
    # ... 其他特征

class PredictionOutput(BaseModel):
    prediction: float # 或合适的类型

# 使用 Depends 注入模型
@app.post("/predict/", response_model=PredictionOutput)
async def make_prediction(
    input_data: PredictionInput,
    model: Any = Depends(get_model) # 在此处注入模型
):
    """
    接收输入数据，使用注入的模型进行预测，
    并返回结果。
    """
    try:
        # 将 Pydantic 模型转换为模型期望的格式
        # 这一步很大程度上取决于你的具体模型
        model_input = [[input_data.feature1, input_data.feature2]] # 示例转换

        # 使用注入的模型进行预测
        prediction_result = model.predict(model_input)

        # 假设预测结果是单个值或易于提取的
        return PredictionOutput(prediction=float(prediction_result[0]))
    except Exception as e:
        # 处理预测过程中可能出现的错误
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
```

在此示例中，当传入请求执行`make_prediction`之前，FastAPI会调用`get_model()`。返回值（我们的`loaded_model`）随后作为`model`参数 (parameter)传递给`make_prediction`。端点函数本身无需了解模型*如何*或*从何而来*；它只是将其作为一个参数接收。

### 为模型使用依赖注入的好处

1. **解耦：** 你的端点逻辑（`make_prediction`）与模型加载逻辑（`get_model`）是解耦的。端点仅关注验证、预测和响应格式化。
2. **可测试性：** 这是一个很大的优势。在为`make_prediction`端点编写单元测试时，你可以轻松提供一个*不同的*依赖函数（一个“模拟”或“假”的），它返回一个可预测的虚拟模型，而不是真实模型。这使得你可以在隔离环境中测试端点逻辑，而无需实际的（可能缓慢或复杂的）模型。FastAPI的测试工具（`TestClient`）与覆盖依赖项功能可以平稳地结合使用。
3. **可重用性：** 如果多个端点都需要访问同一个模型实例，`get_model`依赖函数可以被它们重用。
4. **可维护性：** 如果你需要更改模型加载或管理方式（例如，切换到不同的加载机制或添加缓存），你只需修改`get_model`函数。只要依赖函数继续返回兼容的模型对象，依赖它的端点函数都不需要更改。
5. **清晰度：** 它使得端点的要求在其函数签名中明确体现。任何阅读代码的人都可以立即看到`make_prediction`函数需要一个`model`对象来执行其任务。

虽然将模型加载到一个简单的全局变量中对于非常小的应用来说可能已足够，但使用FastAPI的依赖注入系统为将机器学习 (machine learning)模型集成到生产就绪的API中提供了更清晰、更具扩展性且可测试性更好的架构。这在FastAPI开发中是管理数据库连接、外部服务客户端以及（对我们来说很重要的）机器学习模型等资源的常见做法。

## 参考资料

- [Dependencies in FastAPI](https://fastapi.tiangolo.com/tutorial/dependencies/) — Sebastián Ramírez (2024)
  官方文档，解释了FastAPI的依赖注入系统及其使用方法。
- [Dependency Injection: Principles, Practices, Patterns](https://www.manning.com/books/dependency-injection-principles-practices-patterns-2nd-edition) — Mark Seemann, Steven van Deursen (2019)
  Publisher: Manning Publications
  一本关于依赖注入设计模式、其原则以及在软件工程中实际应用的综合性书籍。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  涵盖设计和构建稳健机器学习系统的策略，包括模型部署和管理方面的考量。
- [Testing FastAPI Applications](https://fastapi.tiangolo.com/tutorial/testing/) — Sebastián Ramírez (2024)
  官方文档，详细说明了如何有效测试FastAPI应用程序，包括如何为测试目的覆盖依赖项。
