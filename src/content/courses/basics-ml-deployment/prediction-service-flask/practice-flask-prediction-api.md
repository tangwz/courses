---
course: "basics-ml-deployment"
chapter: "prediction-service-flask"
lesson: "practice-flask-prediction-api"
sourceId: 4001
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/practice-flask-prediction-api"
title: "动手实践：构建一个简单的Flask预测API"
description: "按照分步教程，创建一个可用的Flask API，用于从您已保存的模型提供预测。"
order: 10
plots: []
sourceHash: "662719e461f4ec2e30044c5d9d48833202a215294bd7eed1d50e6e3c76e7f23c"
sourceCorrections: []
---

构建可用的预测服务结合了API、Flask Web框架以及在Python应用中加载已保存机器学习 (machine learning)模型的过程。详细的指导将引导此类服务的创建。

这个动手练习将逐步指导您使用Flask创建一个简单的Web API。这个API将加载一个预训练 (pre-training)模型（比如您在第二章中保存的模型），并提供一个端点，该端点通过HTTP接收输入数据，使用模型进行预测，并返回结果。

### 前提条件

在开始编码之前，请确保您已准备好以下各项：

1. **Python：** 确保您的系统上已安装Python 3。
2. **Flask：** 您需要安装Flask库。如果尚未安装，请打开您的终端或命令提示符并运行：

   ```bash
   pip install Flask joblib scikit-learn pandas
   ```

   *(我们包含`joblib`、`scikit-learn`和`pandas`，因为它们常用于保存/加载模型和处理数据，如果您的模型使用不同的库如`pickle`，请进行调整)*。
3. **已保存模型：** 您需要一个使用`joblib`或`pickle`保存到文件的训练好的机器学习 (machine learning)模型。对于本例，我们假设您有一个名为`model.joblib`的模型文件，它保存在您将创建Flask应用的同一目录中。这个模型应该经过训练，能够根据特定的输入特征进行预测。我们还将假设这个模型需要以Pandas DataFrame的形式作为输入。
4. **（可选）已保存的预处理器：** 如果您的模型需要单独保存的特定预处理步骤（如缩放或编码）（例如，`scaler.joblib`或完整的`pipeline.joblib`），也请准备好该文件。为了本例的简单起见，我们假设模型文件包含必要的步骤，或者预处理足够简单，可以直接在Flask应用中完成。

### 项目结构

让我们保持文件整洁。为您的项目创建一个新目录，例如`simple_ml_api`。在这个目录中，放置您保存的模型文件（`model.joblib`）。您将在同一目录中创建您的Flask应用脚本，命名为`app.py`。

```
simple_ml_api/
├── app.py          # 您的Flask应用代码
└── model.joblib    # 您保存的模型文件
```

### 步骤1：创建基本的Flask应用（`app.py`）

在您的项目目录中打开一个名为`app.py`的新文件，首先导入必要的库并初始化Flask：

```python
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 初始化Flask应用
app = Flask(__name__)

# 加载训练好的模型（如果您有预处理器也一并加载）
# 我们在应用启动时加载一次，而不是在路由函数内部
# 这样做更高效，因为它避免了每次请求都重新加载。
try:
    model = joblib.load("model.joblib")
    print("模型加载成功。")
    # 如果您有单独的预处理器，请在此处加载：
    # preprocessor = joblib.load("preprocessor.joblib")
except FileNotFoundError:
    print("错误：未找到model.joblib。请确保模型文件位于正确的目录中。")
    model = None
except Exception as e:
    print(f"加载模型时出错：{e}")
    model = None
```

这里，我们导入`Flask`用于Web服务器，`request`用于处理传入数据，`jsonify`用于创建JSON响应，`joblib`用于加载模型，以及`pandas`因为许多模型期望数据以DataFrame格式。我们使用`app = Flask(__name__)`初始化Flask应用。

最重要的是，我们在初始化应用后立即加载`model.joblib`文件。这意味着模型在应用启动时只加载到内存一次。如果在预测函数内部加载，效率会非常低，因为它会为每一个预测请求重新从磁盘加载模型。我们还添加了基本的错误处理，以防模型文件丢失或无法加载。

### 步骤2：定义预测端点

现在，我们来创建处理预测请求的特定URL端点。我们将使用`/predict`路由，并指定它应接受HTTP POST请求，因为客户端将向其*发送*数据。

```python
# 将此代码添加到app.py中模型加载代码的下方

@app.route('/predict', methods=['POST'])
def predict():
    # 检查模型是否加载成功
    if model is None:
        return jsonify({"error": "Model not loaded or failed to load."}), 500

    # 1. 从POST请求中获取数据
    try:
        data = request.get_json(force=True)
        print(f"收到的数据：{data}") # 记录收到的数据

        # 确保数据是预期格式（例如，一个字典）
        if not isinstance(data, dict):
            raise ValueError("Input data must be a JSON object (dictionary).")

        # 2. 为模型准备数据
        # 假设模型期望一个带有特定列名的Pandas DataFrame
        # 根据您模型的训练数据调整列名
        # 示例：{'特征1': 值1, '特征2': 值2, ...}
        feature_values = list(data.values())
        feature_names = list(data.keys()) # 或者明确定义期望的列
        input_df = pd.DataFrame([feature_values], columns=feature_names)

        print(f"准备好的DataFrame：\n{input_df}") # 记录DataFrame

        # 如果您有预处理器，您将在此处应用它：
        # input_processed = preprocessor.transform(input_df)
        # prediction = model.predict(input_processed)

        # 3. 进行预测
        prediction = model.predict(input_df)

        # 如果需要，将预测转换为标准的Python类型（例如，从numpy数组）
        # 确保输出可JSON序列化
        output = prediction[0]
        if hasattr(output, 'item'): # 处理numpy类型
             output = output.item()

        print(f"预测结果：{output}") # 记录预测结果

        # 4. 将预测作为JSON响应返回
        return jsonify({"prediction": output})

    except ValueError as ve:
        print(f"值错误：{ve}")
        return jsonify({"error": f"Invalid input data format: {ve}"}), 400
    except KeyError as ke:
         print(f"错误：{ke}")
         return jsonify({"error": f"Missing expected feature in input data: {ke}"}), 400
    except Exception as e:
        # 捕获处理或预测过程中可能发生的其他错误
        print(f"发生错误：{e}")
        return jsonify({"error": "An error occurred during prediction."}), 500
```

让我们分解一下`predict`函数：

1. **检查模型：** 首先，它验证`model`在启动时是否加载成功。如果没有，则返回错误。
2. **获取数据：** `request.get_json(force=True)`尝试将传入的请求正文解析为JSON。`force=True`有助于在内容类型未明确设置为`application/json`时进行解析，但客户端设置该类型是一个好习惯。我们添加了基本验证，检查接收到的数据是否为字典。
3. **准备数据：** 这一步非常重要，完全取决于您的模型是如何训练的。许多scikit-learn模型期望输入是二维数组状结构（如Pandas DataFrame或NumPy数组），并带有特定顺序的特征列。这里，我们假设输入JSON是一个字典，如`{"feature1": 1.0, "feature2": 2.5, ...}`。我们将其转换为单行Pandas DataFrame。**您必须调整列名和数据准备，以符合您的具体模型要求。** 如果您保存了预处理器或管道，您将在此处应用其`transform`方法。
4. **预测：** 我们调用`model.predict()`方法，传入准备好的数据（`input_df`）。
5. **格式化输出：** 预测结果通常以NumPy数组的形式返回（即使是单个预测）。我们提取第一个元素（`prediction[0]`），并在必要时使用`.item()`将其转换为标准Python类型，确保它可以轻松转换为JSON。
6. **返回响应：** `jsonify({"prediction": output})`创建一个包含预测结果的JSON响应。
7. **错误处理：** `try...except`块会捕获潜在问题，例如缺失JSON数据、数据格式不正确（例如，缺少特征）或预测步骤本身中的错误，并返回带有相应HTTP状态码（客户端错误为400，服务器错误为500）的信息性JSON错误消息。

### 步骤3：添加运行服务器的代码

最后，添加标准的Python构造，使脚本可运行并启动Flask开发服务器：

```python
# 将此代码添加到app.py的末尾

if __name__ == '__main__':
    # 设置host='0.0.0.0'以使服务器可从网络上的其他设备访问
    # 如果需要，使用与默认5000不同的端口
    app.run(host='0.0.0.0', port=5000, debug=True)
```

这段代码检查脚本是否被直接执行（而不是被导入）。`app.run()`启动Flask开发服务器。

- `host='0.0.0.0'`使服务器监听所有可用的网络接口，而不仅仅是localhost。如果您想从网络上的其他设备进行测试，或最终在容器中运行，这会很有用。
- `port=5000`指定端口号（5000是Flask的默认端口）。
- `debug=True`启用调试模式。这会在浏览器中提供更详细的错误消息，并在您保存代码更改时自动重启服务器。**重要：** 由于安全风险和性能开销，请勿在生产环境中使用`debug=True`。

### 步骤4：运行您的Flask API

现在您已准备好运行您的预测服务！

1. 打开您的终端或命令提示符。
2. 导航到您的项目目录（`simple_ml_api`）。
3. 确保您的`model.joblib`文件存在。
4. 运行应用：

   ```bash
   python app.py
   ```

您应该会看到输出，表明模型已成功加载，并且Flask服务器正在运行，通常类似于：

```
模型加载成功。
 * 正在提供Flask应用 'app'
 * 调试模式：开启
警告：这是一个开发服务器。请勿在生产部署中使用它。请改用生产WSGI服务器。
 * 运行在所有地址 (0.0.0.0)
 * 运行在 http://127.0.0.1:5000
 * 运行在 http://[您的本地IP]:5000
按CTRL+C退出
 * 使用stat重启
模型加载成功。
 * 调试器已激活！
 * 调试器PIN码：...
```

您的API现在已运行并在端口5000监听请求！

### 步骤5：测试您的API

您需要一种方式向正在运行的API发送带有JSON数据的POST请求。您可以使用`curl`（一个命令行工具）或使用`requests`库编写一个简单的Python脚本。

**示例输入数据：**

我们假设您的`model.joblib`期望四个特征，分别为`sepal_length`、`sepal_width`、`petal_length`和`petal_width`。您的输入JSON应如下所示：

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

**使用`curl`测试：**

打开*另一个*终端窗口（让第一个窗口继续运行服务器）并执行以下命令。如果需要，替换特征值。

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
     http://127.0.0.1:5000/predict
```

- `-X POST`：将HTTP方法指定为POST。
- `-H "Content-Type: application/json"`：告诉服务器请求体包含JSON数据。
- `-d '{...}'`：在请求体中提供JSON数据。
- `http://127.0.0.1:5000/predict`：您的API端点URL。

**使用Python `requests`测试：**

或者，您可以创建一个小的Python脚本（例如，`test_api.py`）或使用交互式Python会话：

```python
import requests
import json

# 您的Flask API端点URL
url = 'http://127.0.0.1:5000/predict'

# 输入数据，Python字典格式
# 根据您的模型调整特征名称和值
input_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# 发送带有JSON数据的POST请求
response = requests.post(url, json=input_data)

# 检查请求是否成功（状态码200）
if response.status_code == 200:
    # 打印API的JSON响应（预测）
    result = response.json()
    print(f"API响应: {result}")
    # 示例输出可能为：API响应：{'prediction': 0} 或 {'prediction': 'setosa'}
else:
    # 如果请求失败，打印错误信息
    print(f"错误: {response.status_code}")
    try:
        print(f"错误详情: {response.json()}")
    except json.JSONDecodeError:
        print(f"错误详情: {response.text}")
```

运行此脚本（`python test_api.py`）。

**预期输出：**

如果一切正常，`curl`和Python脚本都应收到来自您的API的JSON响应，类似于以下内容（实际预测值取决于您的模型）：

```json
{
  "prediction": 0
}
```

或者（如果您的模型预测类别名称）：

```json
{
  "prediction": "setosa"
}
```

您还应该在运行`app.py`的终端中看到日志消息，显示接收到的数据、准备好的DataFrame和预测结果。

> 流程图显示客户端向正在运行的Flask应用的`/predict`端点发送带有JSON数据的POST请求。应用处理数据，使用加载的模型进行预测，并将结果作为JSON响应返回给客户端。

### 完整的`app.py`示例代码

以下是`app.py`的完整代码，以便查阅：

```python
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 初始化Flask应用
app = Flask(__name__)

# 加载训练好的模型
try:
    model = joblib.load("model.joblib")
    print("模型加载成功。")
except FileNotFoundError:
    print("错误：未找到model.joblib。")
    model = None
except Exception as e:
    print(f"加载模型时出错：{e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    # 检查模型是否加载成功
    if model is None:
        return jsonify({"error": "Model not loaded or failed to load."}), 500

    # 1. 从POST请求中获取数据
    try:
        data = request.get_json(force=True)
        print(f"收到的数据：{data}")

        if not isinstance(data, dict):
             raise ValueError("Input data must be a JSON object (dictionary).")

        # 2. 为模型准备数据
        # 重要提示：调整特征名称以匹配您模型的训练数据
        feature_values = list(data.values())
        feature_names = list(data.keys()) # 或者使用预定义的列表：['sepal_length', 'sepal_width', ...]
        input_df = pd.DataFrame([feature_values], columns=feature_names)

        print(f"准备好的DataFrame：\n{input_df}")

        # 3. 进行预测
        prediction = model.predict(input_df)

        # 4. 格式化输出
        output = prediction[0]
        if hasattr(output, 'item'): # 处理numpy类型
             output = output.item()

        print(f"预测结果：{output}")

        # 5. 将预测作为JSON响应返回
        return jsonify({"prediction": output})

    except ValueError as ve:
        print(f"值错误：{ve}")
        return jsonify({"error": f"Invalid input data format: {ve}"}), 400
    except KeyError as ke:
         print(f"错误：{ke}")
         return jsonify({"error": f"Missing expected feature in input data: {ke}"}), 400
    except Exception as e:
        print(f"发生错误：{e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    # 运行应用，可在网络上访问，并开启调试模式
    # 记住在生产环境中将debug设置为False
    app.run(host='0.0.0.0', port=5000, debug=True)
```

恭喜！您已成功使用Flask构建了一个基本的机器学习 (machine learning)预测API。该服务将您保存的模型包装在一个Web服务器中，通过HTTP接受输入数据并返回预测结果。这是使您的模型可供其他应用或用户使用的基本模式。在下一章中，我们将讨论如何使用Docker打包此应用，使其更具可移植性且更易于部署。

## 参考资料

- [Flask Documentation](https://flask.palletsprojects.com/) — Pallets (2024)
  Flask Web 框架的官方文档，提供了在 Python 中构建 Web 应用程序和 API 的指南和 API 参考。
- [1.1. Model persistence](https://scikit-learn.org/stable/model_persistence.html) — scikit-learn developers (2024)
  scikit-learn 用户指南的这一部分解释了如何使用 joblib 或 pickle 保存和加载训练好的机器学习模型，这是模型部署的一个步骤。
- [Designing Machine Learning Systems: New Responsibilities and Requirements for Data Scientists, ML Engineers, and Product Owners](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  这本书提供了部署和操作机器学习模型的系统层面视角，涵盖了从数据管道到服务预测等方方面面。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/master/) — Kenneth Reitz, and others (2024)
  Python Requests 库的官方文档，对于发出 HTTP 请求以与 Web API 交互很有用。
