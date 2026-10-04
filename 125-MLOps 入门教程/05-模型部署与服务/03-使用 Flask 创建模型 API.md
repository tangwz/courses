# 使用 Flask 创建模型 API

来源：[原文](https://apxml.com/zh/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving/creating-model-api-flask)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练好的机器学习 (machine learning)模型通常以文件形式保存，且处于孤立状态。为了让它发挥作用，其他应用程序需要一种与之通信的方式，即向其发送新数据并接收预测结果。这种通信是通过将模型包装在应用程序编程接口（API）中实现的。API 定义了软件组件之间交互的标准契约。为了方便通过网络与模型交互，构建 Web API 是一种常见的方法，它允许应用程序使用 HTTP 等标准网络协议访问预测功能。

这种方法实现了模型与使用它的应用程序之间的解耦。无论应用程序是移动应用、Web 仪表板还是另一个后端服务，它都不需要了解 Python 或 scikit-learn 的任何信息。它只需要知道如何向特定的 URL（端点）发送 HTTP 请求以及如何解析响应。

### Flask 简介

为了构建这个 Web API，我们将使用 Flask，这是一个流行的 Python “微” Web 框架。“微”并不意味着功能缺失，而是指 Flask 旨在保持核心简单且易于扩展。它提供了构建 Web 应用程序和 API 的基本工具，而不会强加过多的结构或依赖关系。这使得它成为创建轻量级服务的理想选择，这类服务的主要任务是提供模型预测。

在开始之前，你需要安装 Flask。你可以使用 `pip` 来完成：

```bash
pip install Flask scikit-learn joblib
```

我们包含了 `scikit-learn` 和 `joblib`，因为我们将加载一个以该格式保存的简单模型。

### 预测 API 的结构

从本质上讲，一个 Flask API 由几个主要部分组成：

1. **应用实例**：协调整个应用程序的核心对象。
2. **路由**：API 公开的 URL 路径，例如 `/predict` 或 `/health`。
3. **视图函数**：当客户端向特定路由发送请求时执行的 Python 函数。这些函数包含了处理请求和生成响应的逻辑。

让我们先构建一个最简单的 Flask 应用程序，看看这些部分是如何运作的。

```python
# app.py
from flask import Flask

# 1. 创建 Flask 应用实例
app = Flask(__name__)

# 2. 定义路由及其对应的视图函数
@app.route("/")
def index():
    # 3. 该函数返回响应内容
    return "模型 API 正在运行。"
```

在这段代码中，`@app.route("/")` 是一个 Python 装饰器，它告诉 Flask 每当有针对根 URL (`/`) 的 Web 请求时，都应触发 `index()` 函数。

### 加载模型并提供预测

现在，让我们将这个基础结构演变为一个功能齐全的预测服务。一个关于性能的考量是在应用程序启动时仅将模型加载到内存中一次，而不是在每次收到预测请求时都加载。从磁盘加载模型可能是一个耗时的操作，重复执行会造成严重的性能瓶颈。

我们将创建一个新的端点 `/predict`，它通过 HTTP `POST` 请求接收数据。输入数据将采用 JSON 格式，这是在网络上发送结构化数据的标准格式。

以下是一个简单预测 API 的完整代码。假设你有一个训练好的 scikit-learn 模型，保存为 `model.joblib`。

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# 创建 Flask 应用实例
app = Flask(__name__)

# 加载训练好的机器学习模型
# 这在应用程序启动时执行一次
model = joblib.load("model.joblib")

@app.route("/")
def index():
    return "模型 API 正在运行。"

@app.route("/predict", methods=['POST'])
def predict():
    # 从请求中获取 JSON 数据
    data = request.get_json()

    # 基础验证
    if not data or 'features' not in data:
        return jsonify({"error": "无效输入：缺少 'features'。"}), 400

    try:
        # 提取特征并转换为模型所需的 NumPy 数组格式
        features = np.array(data['features']).reshape(1, -1)

        # 使用模型进行预测
        prediction = model.predict(features)

        # 将预测结果转换为标准 Python 类型
        output = prediction.tolist()

        # 将预测结果作为 JSON 响应返回
        return jsonify({"prediction": output})

    except Exception as e:
        # 处理预测过程中可能出现的错误
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 在 0.0.0.0 上运行应用，使其可以从容器外部访问
    app.run(host='0.0.0.0', port=5000)
```

让我们分析一下 `/predict` 函数：

- `methods=['POST']`：这指定了该端点仅响应 `POST` 请求，这是向服务器发送数据以创建或更新资源的常用方法。
- `request.get_json()`：Flask 的 `request` 对象允许我们访问传入的 HTTP 请求。`get_json()` 方法将请求体解析为 JSON 并将其作为 Python 字典返回。
- `np.array(data['features']).reshape(1, -1)`：我们从输入的 JSON 中提取特征列表，并将其转换为 scikit-learn 模型预期的二维 NumPy 数组格式。
- `model.predict(features)`：这是我们使用预先加载的模型进行实际预测的地方。
- `jsonify({"prediction": output})`：我们将预测结果包装在一个字典中，并使用 Flask 的 `jsonify` 工具将其格式化为带有正确 HTTP 标头的 JSON 响应。

下面的图表说明了请求通过 API 的流程。

> 客户端应用程序将特征以 JSON 负载的形式发送到 Flask API。API 使用加载的模型生成预测，并以 JSON 形式返回给客户端。

### 测试模型 API

将代码保存为 `app.py` 后，你可以从终端运行它：

```bash
python app.py
```

你应该会看到显示服务器正在运行的输出，通常在 5000 端口。现在你的 API 正在监听请求。你可以从另一个终端使用 `curl` 等命令行工具进行测试。以下命令向 `/predict` 端点发送包含特征向量 (vector)的 JSON 负载。

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

如果一切运行正常，API 将返回一个包含模型预测结果的 JSON 对象：

```json
{
  "prediction": [0]
}
```

这个简单的 API 是模型部署的基础单元。通过像上一节那样将这个 Flask 应用程序容器化，我们创建了一个可移植、隔离且可扩展的部署单元。然后，该单元可以部署在各种生产环境中，为任何能够进行 HTTP 通信的应用程序提供预测服务。

## 参考资料

- [Flask Documentation](https://flask.palletsprojects.com/en/latest/) — Pallets (2024)
  Flask 微型 web 框架的官方综合指南，对于理解其核心概念以及如何构建 web 应用程序和 API 至关重要。
- [Flask Web Development: Developing Applications with Python](https://www.oreilly.com/library/view/flask-web-development/9781491991721/) — Miguel Grinberg (2018)
  Publisher: O'Reilly Media
  一本引导读者构建 Flask 应用程序的实用书籍，涵盖了 API 开发、请求处理和部署考量，适合初学者和有经验的开发者。
- [Introducing MLOps: How to Go from Model to Production](https://www.oreilly.com/library/view/introducing-mlops/9781492083281/) — Mark Treveil and Dataiku (2020)
  Publisher: O'Reilly Media
  本书概述了 MLOps 的原理和实践，将通过 API 进行模型部署置于更广泛的机器学习生命周期中。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.barnesandnoble.com/w/building-machine-learning-powered-applications-emmanuel-ameisen/1136427515) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media
  提供将机器学习模型转化为可用于生产的应用程序的实用策略，包括关于 API 设计和模型服务的详细讨论。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20Docker%20%E8%BF%9B%E8%A1%8C%E5%AE%B9%E5%99%A8%E5%8C%96.md) · [下一节](04-%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%BC%8F%EF%BC%9A%E8%81%94%E6%9C%BA%E9%A2%84%E6%B5%8B%E4%B8%8E%E6%89%B9%E9%87%8F%E9%A2%84%E6%B5%8B.md)
