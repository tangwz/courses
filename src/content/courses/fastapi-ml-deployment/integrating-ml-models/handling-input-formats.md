---
course: "fastapi-ml-deployment"
chapter: "integrating-ml-models"
lesson: "handling-input-formats"
sourceId: 5280
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/handling-input-formats"
title: "处理不同输入格式"
description: "调整端点以处理各种输入数据类型，例如 JSON 负载或表单数据。"
order: 4
plots: []
sourceHash: "1fd8ac38ce17ff5437a11db2194cfd8c5a1ab6c4f754792bd47a05ee90d751c1"
sourceCorrections: []
---

机器学习 (machine learning)模型通常在特定格式的数据上进行训练，例如 NumPy 数组、Pandas DataFrames 或具有精确形状和数据类型的张量。然而，与您的API交互的客户端通常以更适合网络的格式发送数据，最常见的是JSON负载，偶尔是上传文件。您的FastAPI应用充当桥梁，接收一种格式的数据，并将其转换为加载模型所需的结构。

基于使用 Pydantic 定义的数据验证结构（第2章已讲解），本节着重介绍在您的端点函数中处理这些输入格式的实际步骤，并在调用模型的预测方法前执行必要的预处理。

### 处理 JSON 负载

最常见的情况是接收作为JSON对象或对象数组的输入特征。Pydantic 模型在验证这种传入JSON的结构和类型方面表现出色。

#### 单一预测请求

对于单一实例的预测，客户端通常发送一个JSON对象，其中键代表特征名称，值代表相应的特征值。

```python
# schemas.py
from pydantic import BaseModel

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    # 如果需要，可以添加分类特征的示例：
    # species_guess: str | None = None
```

在您的端点中，您将接收到此 Pydantic 模型的一个实例。您的任务是将此对象转换为模型所需的数值格式，通常是一个2D NumPy数组，其中每行是一个样本（即使只有一个样本）。

```python
# main.py
from fastapi import FastAPI, Depends
import numpy as np
# 假设 schemas.py 包含 IrisFeatures
from .schemas import IrisFeatures
# 假设 model_loader.py 提供一个已加载的模型
from .model_loader import get_model

app = FastAPI()
# 已加载模型对象的占位符（例如 scikit-learn）
# 在实际应用中，请使用后续展示的依赖注入方式
# model = load_my_sklearn_model('path/to/model.joblib')

@app.post("/predict/single")
async def predict_single(
    features: IrisFeatures,
    model = Depends(get_model) # 使用依赖注入
):
    """通过 JSON 接收单个特征集，返回预测结果。"""
    # 如果需要，将 Pydantic 模型转换为列表或字典
    feature_values = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]

    # 转换为 NumPy 数组，确保形状为 (1, num_features)
    # 模型通常期望2D数组，即使是单个样本。
    input_array = np.array(feature_values).reshape(1, -1)

    # 执行预测
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array) # 如果适用

    # 将 NumPy 类型转换为标准 Python 类型以用于 JSON 响应
    return {
        "prediction": prediction[0].item(),
        "probability": probability[0].tolist() # 将数组转换为列表
    }
```

请注意转换为 NumPy 数组以及 `reshape(1, -1)` 调用。这明确创建了一个单行的2D数组，这是许多库（如 scikit-learn）的标准输入格式，即使是预测单个实例时也是如此。此外，请注意在返回 JSON 响应之前，将 NumPy 结果（`prediction[0]`，`probability[0]`）转换回标准 Python 类型（`item()`，`tolist()`）。

#### 批量预测请求

为了提高效率，您可能希望允许客户端在一次 API 调用中发送多个实例进行预测。这通常通过发送一个JSON对象数组来实现。Pydantic 使用 `List[YourModel]` 来处理这种情况。

```python
# schemas.py
from pydantic import BaseModel
from typing import List

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchPredictionRequest(BaseModel):
    instances: List[IrisFeatures]

class PredictionResult(BaseModel):
    prediction: int # 或 float，取决于模型输出
    probability: List[float] | None = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
```

端点会遍历列表，准备每个实例，并将它们收集成一个批次（通常是一个2D NumPy数组），如果模型支持批量推理 (inference)（大多数都支持），则将整个批次提供给模型。

```python
# main.py
# ... 其他导入
from typing import List
from .schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResult

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model = Depends(get_model)
):
    """通过 JSON 数组接收批量特征，返回批量预测结果。"""
    batch_features = []
    for features in request.instances:
        feature_values = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
        batch_features.append(feature_values)

    # 将列表的列表转换为2D NumPy 数组
    input_array = np.array(batch_features)

    # 执行批量预测
    predictions = model.predict(input_array)
    probabilities = model.predict_proba(input_array) # 如果适用

    # 格式化结果
    results = []
    for i in range(len(predictions)):
        results.append(
            PredictionResult(
                prediction=predictions[i].item(),
                probability=probabilities[i].tolist()
            )
        )

    return BatchPredictionResponse(predictions=results)
```

这种批量处理方法通常比进行多次单独的 API 调用更有效，因为它减少了网络开销，并且可以发挥底层机器学习 (machine learning)库优化的批量推理能力。

### 处理文件上传（例如图片）

对于处理非表格数据（如图像、音频或文档）的模型，客户端可能需要直接上传文件，而不是将数据嵌入 (embedding)到JSON中。FastAPI 使用 `File` 和 `UploadFile` 来处理这种情况。

```python
# main.py
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image # 用于图像处理的 Pillow 库
import io
import numpy as np
# 假设 model_loader 提供一个已加载的图像分类模型
from .model_loader import get_image_model

app = FastAPI()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """加载图像字节，为模型进行预处理。"""
    # 预处理示例：打开，调整大小，转换为 NumPy 数组，归一化
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # 常见预处理步骤（根据您的模型调整）
        img = img.resize((224, 224)) # 调整大小示例
        img_array = np.array(img)
        if img_array.ndim == 2: # 处理灰度图
             img_array = np.stack((img_array,)*3, axis=-1) # 转换为3通道
        img_array = img_array / 255.0 # 归一化到 [0, 1]
        # 添加批次维度（1, 高, 宽, 通道）
        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32) # 确保数据类型正确
    except Exception as e:
        # 处理无效图像格式等错误
        print(f"Error preprocessing image: {e}")
        raise ValueError("Invalid image file or format") from e


@app.post("/predict/image")
async def predict_image(
    image_file: UploadFile = File(...),
    model = Depends(get_image_model) # 图像模型的依赖注入
):
    """接收图像文件，返回预测结果。"""
    contents = await image_file.read()
    try:
        input_tensor = preprocess_image(contents)
    except ValueError as e:
        return {"error": str(e)}
    finally:
        await image_file.close() # 关闭文件很重要

    # 执行预测（模型期望预处理过的输入）
    prediction = model.predict(input_tensor) # 或 model.forward(input_tensor) 等

    # 处理预测输出（例如，获取类别标签）
    # 这很大程度上取决于模型的输出格式
    predicted_class_index = np.argmax(prediction[0])
    # 假设 class_names 是端点可用的列表
    # predicted_label = class_names[predicted_class_index]

    return {
        "filename": image_file.filename,
        "content_type": image_file.content_type,
        "prediction_index": predicted_class_index.item(),
        # "predicted_label": predicted_label
    }
```

在此示例中：

1. 我们使用 `UploadFile = File(...)` 定义端点参数 (parameter) `image_file`。
2. 我们使用 `await image_file.read()` 获取文件内容作为字节（因为文件 I/O 本质上是异步的）。
3. 一个辅助函数 `preprocess_image` 封装了将原始字节转换为模型所需特定张量格式的步骤（调整大小、归一化 (normalization)、添加批次维度、确保正确数据类型）。此函数使用 Pillow 库（`PIL`）进行图像处理。
4. 包含了错误处理，以应对文件可能不是有效图像的情况。
5. 在 `finally` 块中使用 `await image_file.close()` 明确关闭文件，以确保资源被释放。
6. 预处理后的数据被送入模型，结果被格式化为响应。

另一种方法（有时用于较小图像）是将图像数据编码为 JSON 负载中的 Base64 字符串。这避免了多部分表单数据，但会增加负载大小。API 端点随后会将 Base64 字符串解码回字节，然后进行预处理。

```python
# schemas.py
from pydantic import BaseModel

class ImageBase64(BaseModel):
    image_b64: str
    filename: str | None = None

# main.py（端点片段）
import base64

@app.post("/predict/image_base64")
async def predict_image_base64(
    request: ImageBase64,
    model = Depends(get_image_model)
):
    """通过 JSON 接收 Base64 编码图像，返回预测结果。"""
    try:
        image_bytes = base64.b64decode(request.image_b64)
        input_tensor = preprocess_image(image_bytes) # 使用相同的预处理器
    except (ValueError, base64.binascii.Error) as e:
        return {"error": f"Invalid Base64 data or image format: {e}"}

    # ...（与之前相同的预测和响应格式化）...

    prediction = model.predict(input_tensor)
    predicted_class_index = np.argmax(prediction[0])

    return {
        "filename": request.filename or "unknown",
        "prediction_index": predicted_class_index.item(),
    }
```

根据客户端需求、预期文件大小和 API 设计偏好选择输入方法（直接文件上传 vs. JSON 中的 Base64）。文件上传通常更适合较大的二进制数据。

### 处理文本数据

对于自然语言处理（NLP）模型，输入通常是文本，作为字符串在 JSON 负载中发送。

```python
# schemas.py
from pydantic import BaseModel
from typing import List

class TextItem(BaseModel):
    text: str

class TextBatchRequest(BaseModel):
    texts: List[str]

# main.py（端点片段）
# 假设 model_loader 提供一个已加载的 NLP 模型/管道
from .model_loader import get_nlp_model

@app.post("/predict/text")
async def predict_text(
    request: TextItem, # 或 TextBatchRequest 用于批量处理
    model = Depends(get_nlp_model)
):
    """接收文本输入，返回 NLP 模型预测结果（例如情感）。"""
    input_text = request.text # 或 request.texts 用于批量处理

    # 预处理可能涉及分词、向量化等。
    # *如果* 这不是加载模型管道的一部分：
    # processed_input = preprocess_text(input_text)
    # prediction = model.predict(processed_input)

    # *如果* 模型（例如 scikit-learn Pipeline）处理预处理：
    prediction = model.predict([input_text]) # 传入原始文本

    # 格式化预测结果
    # 情感分析示例：
    sentiment_score = prediction[0].item()
    sentiment_label = "positive" if sentiment_score > 0.5 else "negative"

    return {
        "input_text": input_text,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
    }
```

对于文本（有时也包括其他数据类型），一个重要考虑是预处理（如分词 (tokenization)或向量 (vector)化）发生的位置。如果您保存了一个包含 `TfidfVectorizer` 等组件的 scikit-learn `Pipeline`，那么您加载的模型文件已经包含了必要的预处理步骤。在这种情况下，您通常可以直接将原始文本传递给管道的 `predict` 方法。如果预处理不是保存模型的一部分，您必须在调用 `predict` 之前，在 FastAPI 端点（或辅助函数）中实现这些精确的步骤。训练预处理和推理 (inference)预处理之间的一致性对于获得正确结果是绝对必要的。

### 数据流图

以下图表展示了处理不同输入格式时典型的数据流：

> 数据从客户端请求经过 FastAPI 处理，再到模型推理 (inference)和响应。

无论输入格式如何（JSON、文件上传），模式都保持相似：接收请求，验证或读取数据，将其预处理成底层机器学习 (machine learning)模型库期望的精确格式，执行预测，并将结果格式化为 JSON 响应。仔细管理这种转换对于构建功能正常且稳定的机器学习预测 API 是必要的。

## 参考资料

- [FastAPI Documentation: Request Body - Multiple Parameters, File Uploads](https://fastapi.tiangolo.com/tutorial/body-multiple-params/) — Sebastián Ramírez (2024)
  介绍 FastAPI 如何处理多种输入类型，包括经 Pydantic 模型验证的 JSON 请求体和通过文件上传接收的二进制数据。
- [Pydantic Documentation: Models](https://docs.pydantic.dev/latest/usage/models/) — Samuel Colvin (2024)
  详细说明 `BaseModel` 的创建和应用，用于定义数据模式、类型验证以及解析 JSON 输入。
- [NumPy User Guide: The N-dimensional array (ndarray)](https://numpy.org/doc/stable/user/basics.html) — The NumPy Developers (2024)
  描述基础的 `ndarray` 对象及其用于数组构建、重塑和操作的方法，对于为机器学习模型准备数据至关重要。
- [Scikit-learn User Guide: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#pipelines-and-composite-estimators) — The scikit-learn developers (2024)
  提供关于使用 `Pipeline` 对象串联多个处理步骤和评估器的指导，确保机器学习工作流程中数据预处理的一致性。
