---
course: "fastapi-ml-deployment"
chapter: "integrating-ml-models"
lesson: "practice-model-prediction-service"
sourceId: 5290
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/practice-model-prediction-service"
title: "实践：构建模型预测服务"
description: "动手实践，将预训练模型集成到 FastAPI 服务中以提供预测。"
order: 8
plots: []
sourceHash: "95645496d773afe7d704434b4a9934ef65d0dba8b12f6e0409f7489d9c28eb0b"
sourceCorrections: []
---

构建一个简单的 FastAPI 服务，用于加载预训练 (pre-training)的机器学习 (machine learning)模型并公开预测接口。它集成了 Pydantic 数据验证，并应用了模型集成技术。

### 前提条件：一个训练好的模型

首先，你需要一个已保存到文件的训练好的机器学习 (machine learning)模型。在这个例子中，我们假设你有一个使用 scikit-learn 在 Iris 数据集上训练并使用 `joblib` 保存的简单分类器。

如果你手头没有现成的，你可以像这样创建一个基础模型：

```python
# train_save_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 加载 Iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

# 训练一个简单的逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# 保存训练好的模型
model_filename = 'iris_classifier.joblib'
joblib.dump(model, model_filename)

print(f"模型已训练并保存到 {model_filename}")
# 预期输出映射：{0: 'setosa', 1: 'versicolor', 2: 'virginica'}
print(f"目标名称: {list(iris.target_names)}")
```

运行此脚本（`python train_save_model.py`），会在你的项目目录中生成 `iris_classifier.joblib` 文件。这个模型需要四个输入特征：萼片长度、萼片宽度、花瓣长度和花瓣宽度（所有单位均为厘米）。它能预测三种鸢尾花物种之一：setosa、versicolor 或 virginica。

### 项目结构

我们来组织一下服务。创建一个包含以下结构的小型项目目录：

```
fastapi_ml_service/
├── iris_classifier.joblib  # 你的已保存模型文件
├── models.py               # 用于请求/响应的 Pydantic 模型
└── main.py                 # 你的 FastAPI 应用程序代码
```

### 定义数据模型（`models.py`）

我们需要 Pydantic 模型来定义输入数据和预测响应的结构。创建 `models.py` 文件：

```python
# models.py
from pydantic import BaseModel, Field
from typing import List

class IrisFeatures(BaseModel):
    """鸢尾花预测的输入特征。"""
    sepal_length: float = Field(..., gt=0, description="萼片长度，单位厘米")
    sepal_width: float = Field(..., gt=0, description="萼片宽度，单位厘米")
    petal_length: float = Field(..., gt=0, description="花瓣长度，单位厘米")
    petal_width: float = Field(..., gt=0, description="花瓣宽度，单位厘米")

    class Config:
        # 用于 FastAPI 文档的示例
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOut(BaseModel):
    """预测输出的模式。"""
    predicted_class_id: int = Field(..., description="预测类别索引（0、1 或 2）")
    predicted_class_name: str = Field(..., description="预测类别名称（'setosa'、'versicolor'、'virginica'）")
    probabilities: List[float] = Field(..., description="每个类别的概率列表 [setosa, versicolor, virginica]")
```

此处：

- `IrisFeatures` 规定了模型所需的四个输入特征。我们使用 `Field` 来添加验证（必须大于 0）和描述，FastAPI 会使用它们自动生成文档。我们还包含了一个示例载荷。
- `PredictionOut` 规定了响应的结构，包括预测类别索引、对应的名称以及每个类别的概率。

### 构建 FastAPI 应用程序（`main.py`）

现在，我们来编写 `main.py` 中的核心应用程序逻辑。我们将在应用启动时加载模型并创建一个 `/predict` 接口。

```python
# main.py
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from models import IrisFeatures, PredictionOut # 导入 Pydantic 模型

# --- 应用程序设置 ---
app = FastAPI(
    title="鸢尾花预测服务",
    description="一个使用预训练模型预测鸢尾花物种的简单 API。",
    version="0.1.0",
)

# --- 模型加载 ---
# 在应用程序启动时加载模型。
# 对于大型应用程序，请考虑依赖注入。
model_path = "iris_classifier.joblib"
try:
    model = joblib.load(model_path)
    print(f"模型从 {model_path} 加载成功")
    # 根据 Iris 数据集标准顺序定义类别名称
    class_names = ['setosa', 'versicolor', 'virginica']
except FileNotFoundError:
    print(f"错误：模型文件在 {model_path} 未找到")
    model = None # 如果加载失败，将模型设为 None
except Exception as e:
    print(f"加载模型时出错：{e}")
    model = None

# --- API 接口 ---
@app.get("/")
def read_root():
    """根接口，提供基本的 API 信息。"""
    return {"message": "欢迎来到鸢尾花预测 API！"}

@app.post("/predict", response_model=PredictionOut)
async def predict_iris(features: IrisFeatures):
    """
    根据输入特征预测鸢尾花物种。

    接受萼片长度、萼片宽度、花瓣长度和花瓣宽度，
    返回预测类别 ID、类别名称和概率。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载或不可用。")

    # 1. 将输入数据转换为模型预期的格式
    #    （scikit-learn 模型通常需要一个 2D NumPy 数组）
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    # 2. 进行预测
    try:
        prediction_id = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
    except Exception as e:
        # 处理预测过程中可能出现的错误
        raise HTTPException(status_code=500, detail=f"预测错误: {e}")

    # 3. 格式化响应
    predicted_class_index = int(prediction_id[0]) # 获取第一个元素

    if predicted_class_index < 0 or predicted_class_index >= len(class_names):
         raise HTTPException(status_code=500, detail="预测索引超出范围。")

    predicted_class_name = class_names[predicted_class_index]
    prediction_probabilities = probabilities[0].tolist() # 获取第一个（也是唯一一个）输入的概率

    return PredictionOut(
        predicted_class_id=predicted_class_index,
        predicted_class_name=predicted_class_name,
        probabilities=prediction_probabilities
    )

# --- 运行应用程序（可选，用于直接执行） ---
# 通常，你会从命令行使用 Uvicorn 运行它。
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
```

`main.py` 中的步骤：

1. **导入必要的库：** `FastAPI`、`joblib`、`numpy`、`HTTPException` 和我们的 Pydantic 模型。
2. **创建 FastAPI 实例：** 使用标题和描述初始化应用程序。
3. **加载模型：** 在应用程序启动时，使用 `joblib.load()` 加载 `iris_classifier.joblib` 文件。其中包含了基本的错误处理。我们还定义了与模型输出对应的 `class_names`。
4. **规定 `/predict` 接口：**
   - 这是一个 POST 接口，因为我们发送数据来创建预测。
   - 它期望一个与 `IrisFeatures` Pydantic 模型相符的请求体。FastAPI 会自动解析传入的 JSON 并根据此模型进行验证。如果验证失败，FastAPI 会自动返回 422 Unprocessable Entity 错误。
   - 它指定 `response_model=PredictionOut`。FastAPI 使用此参数 (parameter)来验证传出的响应、筛选数据（只返回 `PredictionOut` 中定义的字段）并生成文档。
   - 函数内部：
     - 检查模型是否成功加载。
     - 将 `IrisFeatures` 对象转换为 2D NumPy 数组，以符合 scikit-learn 的 `predict` 和 `predict_proba` 方法的要求。
     - 调用 `model.predict()` 获取类别 ID，并调用 `model.predict_proba()` 获取类别概率。
     - 提取相关的预测结果和概率。
     - 返回 `PredictionOut` 模型的一个实例。FastAPI 会自动将这个 Pydantic 对象转换为 JSON 响应。

### 运行和测试服务

1. **启动服务器：** 在 `fastapi_ml_service` 目录中打开终端并运行：

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   - `main:app`：告诉 Uvicorn 在 `main.py` 文件中找到 `app` 对象。
   - `--reload`：当代码更改时自动重启服务器（在开发过程中很有用）。
   - `--host 0.0.0.0`：使服务器可以从你的网络中的其他机器（或后续的 Docker 容器）访问。
   - `--port 8000`：指定运行端口。
2. **访问文档：** 打开你的网页浏览器并访问 `http://localhost:8000/docs`。你应该会看到 FastAPI 生成的交互式 Swagger UI 文档，其中显示了你的 `/` 和 `/predict` 接口，包括 Pydantic 定义的模式。
3. **使用 `curl` 测试：** 打开另一个终端，向 `/predict` 接口发送一个 POST 请求：

   ```bash
   curl -X 'POST' \
     'http://localhost:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "sepal_length": 5.1,
     "sepal_width": 3.5,
     "petal_length": 1.4,
     "petal_width": 0.2
   }'
   ```

   你应该会收到一个类似于这样的 JSON 响应（概率可能略有不同）：

   ```json
   {
     "predicted_class_id": 0,
     "predicted_class_name": "setosa",
     "probabilities": [0.97..., 0.02..., 0.00...]
   }
   ```
4. **测试验证：** 尝试发送无效数据（例如，缺少字段或提供非数字数据）：

   ```bash
   curl -X 'POST' \
     'http://localhost:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "sepal_length": 5.1,
     "sepal_width": "not-a-number",
     "petal_length": 1.4,
     "petal_width": 0.2
   }'
   ```

   FastAPI（得益于 Pydantic）将自动返回一个 `422 Unprocessable Entity` 错误，其中包含验证失败的详细信息：

   ```json
   {
     "detail": [
       {
         "loc": [
           "body",
           "sepal_width"
         ],
         "msg": "value is not a valid float",
         "type": "type_error.float"
       }
     ]
   }
   ```

你现在已经成功构建了一个可工作的机器学习 (machine learning)预测服务，使用了 FastAPI！它加载模型，通过 Pydantic 为输入和输出规定了清晰的数据约定，并通过一个明确的 API 接口提供预测。这为部署更复杂的模型建立了坚固的根基。在后续章节中，我们将讨论组织大型应用程序、测试、处理异步操作和容器化。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI Web框架的官方指南，涵盖了其构建带有自动数据验证和文档的异步API的特点。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin (2024)
  Pydantic的官方文档，该库使用Python类型提示实现数据验证和设置管理。
- [Scikit-learn User Guide: 1.1. Model persistence](https://scikit-learn.org/stable/modules/model_persistence.html) — scikit-learn developers (2024)
  scikit-learn用户指南的一个章节，解释了如何使用`joblib`或`pickle`保存和加载训练好的机器学习模型。
- [Uvicorn Documentation](https://www.uvicorn.org/) — Encode (Tom Christie and contributors) (2024)
  Uvicorn的官方文档，Uvicorn是Python的ASGI Web服务器实现。
