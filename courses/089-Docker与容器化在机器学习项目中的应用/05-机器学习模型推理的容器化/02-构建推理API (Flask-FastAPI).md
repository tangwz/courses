# 构建推理API (Flask/FastAPI)

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/building-inference-apis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)模型训练并保存后，需要一个接口来接收输入数据并返回预测结果。对于大多数部署情况，简单地按需运行Python脚本是不实际的。通常，我们会将模型加载和预测的逻辑封装在一个Web应用程序编程接口（API）中。这个API作为标准化的入口，允许其他服务或应用通过网络（通常是HTTP请求）请求预测。

对于基于Python的机器学习模型，用于构建这些API的两个流行微框架是Flask和FastAPI。它们提供了处理传入的Web请求、处理数据、与模型交互以及发送响应的工具，且所需样板代码相对较少。

### 为何使用Web框架？

创建API能使您的模型交互标准化。无需依赖自定义脚本或特定的执行环境，任何能发送HTTP请求的人（或应用）都能使用您的模型。这有多个好处：

- **解耦：** 请求预测的服务无需知道模型运行的内部细节，只需了解如何构建请求和解释响应。
- **可访问性：** 模型通过网络变得可访问，这使得分布式系统和微服务架构成为可能。
- **可伸缩性：** Web框架通常与生产级别的Web服务器（例如Flask的Gunicorn、FastAPI的Uvicorn）集成，这些服务器可以并发处理多个请求，为可伸缩部署提供了支持。

### Flask：简洁与成熟

Flask是一个轻量级的WSGI（Web服务器网关接口）Web应用框架。它以简洁和核心最小化著称，易于上手。您可以根据需要通过扩展添加功能。

一个用于推理 (inference)的最小Flask应用可能如下所示：

1. **加载模型：** 应用启动时加载序列化模型（例如，来自`.joblib`或`.pkl`文件）。请注意此模型文件的位置，尤其是在容器内运行时（通常在构建期间复制进去或通过卷挂载）。
2. **定义一个端点：** 创建一个路由（例如，`/predict`），监听特定的HTTP方法（通常是`POST`用于发送数据）。
3. **处理输入：** 从传入请求（通常是JSON）中提取数据，对其进行验证，并预处理成模型期望的格式（例如，NumPy数组或Pandas DataFrame）。
4. **获取预测：** 将处理后的数据传递给已加载模型的`predict()`或`predict_proba()`方法。
5. **返回响应：** 将模型输出（例如，预测值、概率）格式化为JSON响应并发送回客户端。

**示例：Scikit-learn模型的Flask API**

我们假设您有一个训练好的Scikit-learn模型，保存为`model.joblib`。

- **`app.py` (Flask)：**

```python
import joblib
from flask import Flask, request, jsonify
import pandas as pd

# 初始化 Flask 应用
app = Flask(__name__)

# 加载训练好的模型
# 确保 'model.joblib' 在容器的工作目录中可访问
# 或者通过卷挂载。
try:
    model = joblib.load('model.joblib')
    print("模型加载成功。")
except FileNotFoundError:
    print("错误：未找到 model.joblib。请确保它在正确的路径中。")
    model = None # 优雅处理或退出
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None

# 定义预测端点
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    try:
        # 从请求中获取 JSON 数据
        data = request.get_json(force=True)

        # 基本输入验证（示例：检查 'features' 键）
        if 'features' not in data:
            return jsonify({"error": "JSON 数据中缺少 'features' 键"}), 400

        # 将输入数据转换为 DataFrame（根据您的模型调整列）
        # 这假设输入格式如下: {"features": [[值1, 值2, ...], [值1, 值2, ...]]}
        # 或者: {"features": [{"列1": 值1, "列2": 值2}, {"列1": 值1, "列2": 值2}]}
        input_df = pd.DataFrame(data['features'])

        # 进行预测
        predictions = model.predict(input_df)

        # 将预测结果转换为列表以便 JSON 序列化
        output = predictions.tolist()

        # 以 JSON 格式返回预测结果
        return jsonify({"predictions": output})

    except Exception as e:
        # 基本错误处理
        print(f"预测错误: {e}")
        return jsonify({"error": "预测失败", "details": str(e)}), 500

# 运行应用（用于本地开发）
# 在生产环境中，请使用 Gunicorn 等 WSGI 服务器：
# gunicorn --bind 0.0.0.0:5000 app:app
if __name__ == '__main__':
    # 默认使用端口 5000，或根据需要配置
    app.run(host='0.0.0.0', port=5000, debug=False) # 在生产环境中设置 debug=False
```

- **`requirements.txt`：**

```
flask
pandas
scikit-learn
joblib
# 如果使用 gunicorn 作为服务器，请添加它
gunicorn
```

要在本地运行此应用：`python app.py`。要使用生产服务器运行：`gunicorn --bind 0.0.0.0:5000 app:app`。在Docker容器中，`CMD`或`ENTRYPOINT`通常会调用`gunicorn`。

### FastAPI：性能与现代特性

FastAPI是一个现代、高性能的Web框架，它构建于Starlette（用于Web部分）和Pydantic（用于数据验证）之上。它使用ASGI（异步服务器网关接口），支持异步代码（`async`/`await`），这可以大幅提升Web服务中常见的I/O密集型任务的吞吐量 (throughput)。

FastAPI用于推理 (inference)API的主要优点包括：

- **性能：** 由于其异步特性和底层库（Starlette、Uvicorn），通常比Flask更快。
- **数据验证：** 使用Python类型提示和Pydantic进行自动请求/响应数据验证和序列化，减少样板代码并及早发现错误。
- **自动文档：** 从您的代码自动生成交互式API文档（Swagger UI和ReDoc）。
- **异步支持：** 自然地处理异步操作，如果您的预处理或后处理涉及I/O（例如获取额外数据），这将非常有用。

**示例：Scikit-learn模型的FastAPI API**

使用相同的`model.joblib`：

- **`app.py` (FastAPI)：**

```python
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
import pandas as pd
import uvicorn # 本地运行必需

# 使用 Pydantic 定义输入数据模式
# 选项 1：特征的列表嵌套列表
# class PredictionInput(BaseModel):
#     features: List[List[Union[float, int, str]]]

# 选项 2：特征的字典列表（更明确）
class FeatureRow(BaseModel):
    # 明确定义预期的特征名称和类型
    # 示例：
    feature1: float
    feature2: int
    feature3: str
    # 添加您的模型期望的所有特征...

class PredictionInput(BaseModel):
    features: List[FeatureRow]

# 定义输出数据模式
class PredictionOutput(BaseModel):
    predictions: List[Any] # 如果已知具体类型（int, float, str），请将 'Any' 调整为该类型

# 初始化 FastAPI 应用
app = FastAPI(title="机器学习模型推理API", version="1.0")

# 加载训练好的模型
try:
    model = joblib.load('model.joblib')
    print("模型加载成功。")
except FileNotFoundError:
    print("错误：未找到 model.joblib。请确保它在正确的路径中。")
    model = None
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None

@app.on_event("startup")
async def startup_event():
    # 如果愿意，也可以在此处加载模型
    # 全局 'model' 变量需要可访问
    if model is None:
      print("启动时无法加载模型。")
      # （可选）抛出错误或以不同方式处理

@app.get("/")
def read_root():
    return {"message": "前往机器学习推理API"}

# 定义预测端点
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        # Pydantic 根据 PredictionInput 模式自动验证了输入
        # 将 Pydantic 模型转换为字典列表，然后转换为 DataFrame
        input_dict_list = [row.dict() for row in input_data.features]
        input_df = pd.DataFrame(input_dict_list)

        # 如有必要，确保列顺序与模型训练时一致
        # input_df = input_df[expected_column_order]

        # 进行预测
        predictions = model.predict(input_df)

        # 将预测结果转换为列表
        output = predictions.tolist()

        # 返回预测结果（FastAPI 对 PredictionOutput 进行验证）
        return PredictionOutput(predictions=output)

    except Exception as e:
        print(f"预测错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")

# 运行应用（用于本地开发）
# 在生产环境中，直接使用 Uvicorn 或通过 Gunicorn 和 Uvicorn workers 运行：
# uvicorn app:app --host 0.0.0.0 --port 8000
# gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
if __name__ == '__main__':
    # FastAPI 默认使用端口 8000
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=False) # 开发时设置为 reload=True
```

- **`requirements.txt`：**

```
fastapi
uvicorn[standard] # 包含性能扩展
pandas
scikit-learn
joblib
pydantic
# 如果使用 gunicorn 作为服务器，请添加它
gunicorn
```

要在本地运行此应用：`uvicorn app:app --reload --port 8000`。对于生产环境：`uvicorn app:app --host 0.0.0.0 --port 8000`。在Docker内部，`CMD`通常会调用`uvicorn`或带有Uvicorn workers的`gunicorn`。在浏览器中访问`http://localhost:8000/docs`将显示自动生成的交互式文档。

### 与Docker集成

无论您选择Flask还是FastAPI，将API容器化的过程都是相似的：

1. **Dockerfile：** 从合适的`基础镜像`（例如`python:3.9-slim`）创建`Dockerfile`。
2. **安装依赖：** 复制您的`requirements.txt`文件并运行`pip install -r requirements.txt`。
3. **复制应用代码：** 复制您的`app.py`和任何其他必要的脚本或模块。
4. **复制模型文件：** 将`model.joblib`（或其他模型文件）复制到镜像中。或者，如果模型较大或经常更新，则可以计划使用卷进行挂载。
5. **暴露端口：** 使用`EXPOSE`指令来表明您的应用监听的端口（例如，Flask的`EXPOSE 5000`，FastAPI的`EXPOSE 8000`）。请注意，`EXPOSE`仅为文档说明；在运行容器时仍需使用`-p`发布端口（`docker run -p 8080:8000 ...`）。
6. **定义命令：** 使用`CMD`或`ENTRYPOINT`来指定启动Web服务器的命令（例如，`CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]` 或 `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`）。使用`0.0.0.0`作为主机可以确保服务器监听容器内部所有可用的网络接口，从而在端口发布后可以从外部访问。

使用Flask或FastAPI构建API提供了一种服务机器学习 (machine learning)模型的标准方式。两者之间的选择通常取决于项目要求：Flask的简洁性非常适合小型项目或性能不是主要考虑因素的情况，而FastAPI的性能、内置验证和异步能力使其对要求更高的应用具有吸引力。两者都与Docker良好集成，可用于创建便携和可伸缩的推理 (inference)服务。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI的官方综合指南，详细介绍了其构建高性能API的功能，包括自动数据验证和文档生成。
- [Flask Documentation](https://flask.palletsprojects.com/en/latest/) — Pallets (2024)
  Flask的官方指南，提供其核心概念以及如何开发Web应用程序和API的信息，包括请求和响应处理。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建、部署和维护机器学习系统的端到端流程的书，包括服务推理API的架构模式。
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/) — Samuel Colvin (2024)
  Pydantic的官方文档，Pydantic是一个数据验证和设置管理库，FastAPI使用它来定义清晰且可靠的输入/输出模式。
- [Uvicorn Documentation](https://www.uvicorn.org/) — encode (2024)
  Publisher: encode
  Uvicorn的官方文档，Uvicorn是一个ASGI Web服务器，解释了它在高效服务异步Python Web应用程序（如用FastAPI开发的应用程序）中的作用。

---

[上一节](01-%E8%AE%BE%E8%AE%A1%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1.md) · [下一节](03-%E4%BC%98%E5%8C%96%E9%95%9C%E5%83%8F%E5%A4%A7%E5%B0%8F%EF%BC%9A%E5%A4%9A%E9%98%B6%E6%AE%B5%E6%9E%84%E5%BB%BA.md)
