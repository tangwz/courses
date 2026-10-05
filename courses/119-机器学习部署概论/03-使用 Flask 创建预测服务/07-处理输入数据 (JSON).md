# 处理输入数据 (JSON)

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/handling-input-json)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Flask应用要进行预测，需要输入数据。当用户或另一个应用想要进行预测时，他们需要将输入特征（如测量值、文本或图像数据）发送到API。接收和处理这些通常以JSON格式呈现的传入数据，是Flask服务的一个主要功能。

### API数据为何选用JSON？

设想您需要在互联网上传输结构化信息。您需要一种既方便人类读写，又便于机器解析和生成的格式。JSON（JavaScript对象表示法）完美符合这一描述。

JSON以键值对（类似于Python字典）和有序列表（类似于Python列表）的形式表示数据。其简洁性和文本性质使其成为Web API中数据传输的事实标准。

以下是一个简单模型的输入特征在JSON格式下的示例：

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

或者，如果您预期多个特征以列表形式存在：

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

使用JSON为发送请求的客户端和接收请求的Flask API提供了一种通用语言。

### 在Flask中访问请求数据

Flask使得访问传入Web请求中发送的数据变得简单。当客户端通过HTTP请求的*主体*（通常使用`POST`方法）发送数据（如上述JSON）时，Flask提供了一个全局的`request`对象。您需要从`flask`库中导入它：

```python
from flask import Flask, request, jsonify
import joblib # 或者 pickle

# 假设您的Flask应用已初始化
app = Flask(__name__)

# 加载您的模型（根据需要调整路径）
# model = joblib.load('your_model.pkl')
# preprocessor = joblib.load('your_preprocessor.pkl') # 如果您有预处理器的话

@app.route('/predict', methods=['POST'])
def predict():
    # 在此处访问传入的请求数据
    pass # 我们稍后会补全这里
```

在您的路由函数中（本例为`predict`），`request`对象包含传入请求的所有信息，包括头部、参数 (parameter)，以及重要的是，请求主体中发送的数据。

### 解析JSON数据

由于我们预期客户端以JSON格式发送数据，Flask提供了一个便捷方法：`request.get_json()`。此方法尝试将请求主体解析为JSON，并返回一个Python字典（或列表，取决于JSON结构）。

让我们更新`predict`函数以使用它：

```python
# （Flask应用设置和模型加载部分在上面）

@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求的内容类型是否为JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # 从请求主体解析JSON数据
    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}") # 记录接收到的数据
    except Exception as e:
        app.logger.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Could not parse JSON data"}), 400

    # --- 下一步：验证并使用数据 ---
    # （验证和预测的占位符）
    return jsonify({"message": "Data received, processing..."}) # 临时响应
```

以下是具体的步骤：

1. 我们首先检查`request.is_json`。这会查看请求的`Content-Type`头部（例如`application/json`），以确认客户端是否意图发送JSON。如果不是，我们将返回一个HTTP状态码400（Bad Request）的错误响应。
2. 我们使用`request.get_json()`来解析数据。如果数据不是有效的JSON，此方法可能会抛出异常，因此我们将其封装在一个`try...except`块中，用于基本的错误处理。
3. 我们使用`app.logger.info`记录接收到的数据。日志对于调试非常有用。
4. 如果解析失败，我们会记录错误并返回特定的错误消息和400状态码。

### 验证输入数据

仅仅接收JSON是不够的。您的模型期望的数据具有*特定结构*和*格式*。在将数据馈送给模型之前，您应该进行一些验证：

1. **结构检查：** JSON是否包含您预期的键？如果您预期一个包含`features`键的字典，您应该检查该键是否存在。
2. **格式/类型检查（基本）：** 与键关联的值是否为您的模型所需的格式？如果`features`应该是一个数字列表，请检查它是否确实是一个列表，并检查其元素是否看起来像数字。更复杂的类型验证是可行的，但对于一个简单的服务可能不必要。

让我们添加一个简单的结构检查：

```python
# （Flask应用设置和模型加载部分在上面）

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")
    except Exception as e:
        app.logger.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Could not parse JSON data"}), 400

    # --- 基本验证 ---
    # 示例：检查'features'键是否存在且为列表
    if 'features' not in data or not isinstance(data['features'], list):
        app.logger.error("Invalid input: 'features' key missing or not a list")
        return jsonify({"error": "Missing or invalid 'features' key. Expected a list."}), 400

    input_features = data['features']
    # （可选择添加更多检查，例如特征数量）
    app.logger.info(f"Extracted features: {input_features}")

    # --- 下一步：为模型准备数据并进行预测 ---
    # （预测逻辑的占位符）
    return jsonify({"message": f"Received features: {input_features}"}) # 临时响应
```

这一验证步骤有助于避免稍后在使用数据与模型时出现错误，并在客户端发送格式不正确的数据时提供更清晰的错误消息。

### 为模型准备数据

通常，通过JSON接收的数据格式并非完全符合您的机器学习 (machine learning)模型的预期。例如，scikit-learn模型通常要求输入为NumPy数组或类似结构（如列表的列表，其中每个内部列表是一个样本）。

您需要将从JSON中提取的数据转换为所需的格式。如果您还保存了预处理步骤（如缩放器或编码器），您也将在此时应用它们。

```python
# （Flask应用设置、模型加载、以及像numpy这样的导入）
import numpy as np # 确保导入numpy

@app.route('/predict', methods=['POST'])
def predict():
    # （JSON解析和验证如上...）

    # --- 为模型准备数据 ---
    try:
        # 假设'features'是一个数字列表 [f1, f2, f3, ...]
        input_features = data['features']

        # 示例：转换为适合scikit-learn的NumPy数组
        # 模型预期一个2D数组：[[f1, f2, f3, ...]] 用于单个预测
        model_input = np.array(input_features).reshape(1, -1)
        app.logger.info(f"Prepared model input shape: {model_input.shape}")

        # 如果您有预处理器：
        # model_input = preprocessor.transform(model_input)
        # app.logger.info("已应用预处理")

    except Exception as e:
        app.logger.error(f"Error preparing data for model: {e}")
        return jsonify({"error": "Invalid data format for model processing."}), 400

    # --- 下一步：进行预测 ---
    # prediction = model.predict(model_input)
    # （预测逻辑和返回结果的占位符）
    return jsonify({"message": f"Model input prepared: {model_input.tolist()}"}) # 临时响应
```

在此步骤中：

1. 我们提取特征列表。
2. 我们将列表转换为NumPy数组。
3. 我们使用`reshape(1, -1)`，因为大多数scikit-learn模型期望一个2D数组，其中每一行都是一个样本，即使我们只预测一个样本。
4. 一个占位符注释显示了您将应用任何已保存的预处理步骤的位置。
5. 我们包含了错误处理，以防转换失败（例如，如果列表中包含非数字数据）。

现在，您的Flask应用能够接收JSON数据，执行基本验证，并将其转换为可供您的机器学习模型使用的格式。下一步是实际使用这些准备好的数据进行预测并返回结果。

## 参考资料

- [Quickstart](https://flask.palletsprojects.com/en/latest/quickstart/) — Pallets (2024)
  Flask快速入门指南，提供了设置基本Flask应用程序、定义路由和处理Web请求的基础说明，作为API的基础。

---

[上一节](06-%E5%AE%9A%E4%B9%89%E9%A2%84%E6%B5%8B%E7%AB%AF%E7%82%B9.md) · [下一节](08-%E8%BF%94%E5%9B%9E%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
