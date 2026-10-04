---
course: "basics-ml-deployment"
chapter: "prediction-service-flask"
lesson: "testing-api-locally"
sourceId: 3997
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/testing-api-locally"
title: "在本地测试你的 API"
description: "使用 curl 或 Python requests 等工具向你的本地 Flask API 发送数据并验证预测结果。"
order: 9
plots: []
sourceHash: "16a291b7ac17f3bd5baa574a65e15de3983c6825a38d10fbc1e43c4b7a4dd28f"
sourceCorrections: []
---

开发一个加载训练好的机器学习 (machine learning)模型并定义预测服务断点的 Flask 应用程序，是机器学习部署中的一项主要任务。一旦创建了这样的应用程序，确认其操作正确性是必要的。在考虑将此服务打包或部署到其他任何地方之前，必须在此本地机器上进行测试。这种本地测试周期能够帮助你快速找出并解决 API 逻辑、数据处理或模型集成方面的问题。

### 为何在本地测试？

在本地运行你的 Flask 应用程序并发送测试请求有助于确认几点：

1. **连接性：** 你能实际访问正在运行的 Flask 应用程序吗？
2. **端点识别：** 你的应用程序是否正确识别 `/predict` 路由（或你命名的其他名称）？
3. **数据处理：** 你的 Flask 端点能否正确接收输入数据？它是否按预期解析 JSON 有效载荷？
4. **模型交互：** 加载的模型是否成功地使用输入数据生成了预测结果？
5. **响应格式：** 应用程序是否以预期格式（例如 JSON）返回预测结果？

在这个阶段发现错误比部署后排查问题要快得多、简单得多。

### 启动你的 Flask 服务器

首先，确保你的 Flask 应用程序正在运行。打开你的终端或命令提示符，导航到包含你的 Flask 应用程序文件（例如 `app.py`）的目录，然后使用 Python 运行它：

```bash
python app.py
```

你应该会看到服务器已启动的输出，通常会提及它正在 `http://127.0.0.1:5000/` 或 `http://localhost:5000/` 上运行。请保持此终端窗口打开；关闭它将停止你的服务器。

### 发送 HTTP 请求的工具

为了与你正在运行的 API 交互，你需要一种向其发送 HTTP 请求的方法。可以将其视为一个你可以编程控制的专用网页浏览器。为此常用的两种工具是 `curl` 和 Python 的 `requests` 库。

#### 使用 `curl`（命令行）

`curl` 是一个多功能的命令行工具，用于使用 URL 传输数据。它在大多数 Linux 和 macOS 系统上可用，并且可以在 Windows 上安装。

为了测试你的预测端点（假设它位于 `/predict` 并且期望一个带有 JSON 数据的 POST 请求），你会使用类似这样的命令：

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [1.5, 2.8, 0.5]}' \
     http://127.0.0.1:5000/predict
```

让我们来分析一下这条命令：

- `curl`：调用该工具。
- `-X POST`：将 HTTP 方法指定为 POST。这通常用于发送数据以创建或更新资源，或者在我们这里，用于提交数据以进行预测。
- `-H "Content-Type: application/json"`：设置 `Content-Type` 标头。这告诉 Flask 服务器，你请求正文中发送的数据是 JSON 格式的。你的 Flask 端点很可能期望这种格式。
- `-d '{"features": [1.5, 2.8, 0.5]}'`：提供数据有效载荷（请求的主体）。将示例 `{"features": [1.5, 2.8, 0.5]}` 替换为你的特定模型期望的有效 JSON 输入。请注意 JSON 字符串周围的单引号，以防止 shell 解释 JSON 内部双引号时出现问题。
- `http://127.0.0.1:5000/predict`：你的预测端点的 URL。确保地址、端口 (`5000`) 和路由 (`/predict`) 与你正在运行的 Flask 应用程序匹配。

如果成功，`curl` 将把你的 Flask 应用程序返回的响应主体直接打印到你的终端。

#### 使用 Python `requests`

另外，你可以使用流行的 `requests` 库编写一个小型 Python 脚本。如果你的输入数据复杂，或者你想将 API 测试集成到更大的 Python 工作流中，这会更方便。

首先，确保你已安装该库：

```bash
pip install requests
```

然后，你可以创建一个像 `test_api.py` 这样的脚本：

```python
import requests
import json

# 你本地 Flask API 端点的 URL
url = 'http://127.0.0.1:5000/predict'

# 你的模型所需的输入数据，格式应与你的 API 期望的一致
# 确保这与你在 Flask 端点中处理的结构匹配
input_data = {
    'features': [1.5, 2.8, 0.5]
}

# 发送带有 JSON 数据的 POST 请求
# `json` 参数会自动将 Content-Type 标头设置为 application/json
try:
    response = requests.post(url, json=input_data)
    response.raise_for_status() # 对于不良状态码（4xx 或 5xx）抛出异常

    # 打印状态码和响应 JSON
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())

except requests.exceptions.RequestException as e:
    print(f"发送请求错误: {e}")
except json.JSONDecodeError:
    print("错误：响应不是有效的 JSON")
    print("响应文本:", response.text)
```

从你的终端运行此脚本：

```bash
python test_api.py
```

此脚本将输入数据构建为一个 Python 字典，使用 `requests.post` 将其发送到指定的 URL（自动处理 JSON 序列化和设置 `Content-Type` 标头），然后打印从服务器接收到的状态码和 JSON 响应。

### 解释响应

无论你使用 `curl` 还是 `requests`，都需要仔细检查输出：

1. **HTTP 状态码：** 成功的请求通常会返回 `200 OK` 状态码。如果你看到 `404 Not Found`、`400 Bad Request` 或 `500 Internal Server Error` 等代码，这表示存在问题。
2. **响应主体：** 这应该包含你的模型的预测结果，通常格式为 JSON（例如 `{"prediction": 1}` 或 `{"probabilities": [0.1, 0.9]}`）。验证结构和值是否与你期望的相符。

### 常见问题排查

如果第一次尝试不成功，别担心。以下是常见问题以及如何处理它们：

- **连接被拒绝：** 你启动 Flask 服务器了吗？它是否在正确的地址和端口 (`127.0.0.1:5000`) 上运行？检查你启动 `app.py` 的终端。
- **404 未找到：** 仔细检查你发送请求的 URL。你是否正确输入了 `/predict`？它是否与你的 Flask 应用程序中定义的路由 (`@app.route('/predict', ...)` ) 匹配？HTTP 方法是否正确（例如，向期望 POST 的端点发送 POST 请求）？
- **400 错误请求：** 这通常表示服务器无法理解你的请求，经常是因为输入数据格式不正确。
  - 你正在发送 JSON 数据吗？
  - 你是否设置了 `Content-Type: application/json` 标头（如果使用 `curl`）？
  - JSON 结构是否与你的 Flask 端点期望的完全匹配（例如，一个带有特定键如 `'features'` 的字典）？
- **500 内部服务器错误：** 这表示你的 Flask 应用程序在处理请求时发生了错误。检查你的 Flask 服务器 (`app.py`) 运行的终端。Flask 通常会在那里打印详细的错误消息（“traceback”），这将指示你导致问题的代码行。常见原因包括：
  - 模型文件（`.pkl` 或 `.joblib`）未找到或无法加载。
  - 在 Flask 端点内的数据预处理过程中发生错误。
  - 模型的 `predict()` 方法失败（可能是由于不正确的输入形状或数据类型）。
  - 在格式化响应时发生错误。
- **预测不正确：** 如果你得到了响应，但预测值似乎不正确，请追踪数据流：输入数据是否正确接收？预处理逻辑是否与训练期间使用的逻辑相同？是否加载了正确的模型版本？

通过系统地在本地测试你的 API，你将对你的预测服务按预期运行获得信心。一旦你验证它在你的机器上可靠运行，你将为下一步更好地准备，例如使用 Docker 打包你的应用程序。

## 参考资料

- [Requests: HTTP for Humans](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and Contributors (2024)
  Python Requests 库的官方文档，提供了进行各种 HTTP 请求（包括带有 JSON 数据的 POST 请求）的清晰指导，与 API 交互直接相关。
- [curl Man Page](https://curl.se/docs/manpage.html) — Daniel Stenberg and curl contributors (2024)
  curl 命令行工具的详细手册页，详细说明了发送 HTTP 请求的选项，包括指定方法、标头和数据载荷。
