# 实践：创建模型API并将其容器化

来源：[原文](https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals/model-api-deployment-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个已训练、评估并保存的模型需要使其可被访问，以便应用程序可以从中请求预测。创建一个使用Flask为模型开发的简单Web API，并将其打包到Docker容器中，有助于实现便捷的部署和扩展。

我们将假设您有一个已训练好的scikit-learn模型并已保存到文件中（例如`model.joblib`）。在此示例中，我们设想它是一个在具有四个特征的数据上训练的简单分类器，类似于鸢尾花数据集。您还需要在系统上安装Python、pip和Docker。

### 项目设置

首先，为您的项目创建一个目录。在这个目录中，我们将放置应用程序代码、模型文件、依赖项信息和Docker配置。

```bash
mkdir model_api_project
cd model_api_project
# 假设您将训练好的模型复制到此处
# cp path/to/your/model.joblib ./model.joblib
touch app.py requirements.txt Dockerfile
```

您的目录应如下所示：

```
model_api_project/
├── app.py              # Flask 应用程序代码
├── model.joblib        # 您已保存的机器学习模型
├── requirements.txt    # Python 依赖项
└── Dockerfile          # 构建 Docker 镜像的指令
```

### 步骤 1：使用 Flask 创建预测 API

我们将使用 Flask（一个轻量级 Python Web 框架）来创建一个 API 端点，该端点接受数据，使用我们加载的模型进行预测，并返回结果。

编辑 `app.py` 并添加以下代码：

```python
import joblib
import numpy as np
from flask import Flask, request, jsonify

# 1. 加载训练好的模型
# 确保 'model.joblib' 与 app.py 在同一目录中
# 或者提供模型文件的完整路径。
try:
    model = joblib.load('model.joblib')
    print("模型加载成功。")
except FileNotFoundError:
    print("错误：model.joblib 未找到。请确保模型文件在正确的目录中。")
    model = None # 如果加载失败，将模型设为 None
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None # 如果加载失败，将模型设为 None

# 2. 创建 Flask 应用程序实例
app = Flask(__name__)

# 3. 定义预测端点
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未加载，无法进行预测。'}), 500

    try:
        # 从 POST 请求体中获取数据
        data = request.get_json()

        # 验证输入数据结构（基本示例）
        if 'features' not in data or not isinstance(data['features'], list):
             return jsonify({'error': 'JSON payload 中缺少或“features”字段无效。预期为列表。'}), 400

        # 假设模型期望一个用于预测的二维数组结构
        # （例如，对于 scikit-learn 模型）
        # 如果可能，对特征数量进行基本验证
        # 示例：如果您的模型期望 4 个特征
        # if len(data['features']) != 4:
        #    return jsonify({'error': f'预期 4 个特征，得到 {len(data["features"])}'}), 400

        features = np.array(data['features']).reshape(1, -1) # 为单次预测重塑

        # 进行预测
        prediction = model.predict(features)

        # 准备响应
        # 将 numpy 数组类型转换为标准 Python 类型以进行 JSON 序列化
        # 示例：如果预测是数字
        prediction_result = prediction[0].item() if isinstance(prediction[0], np.generic) else prediction[0]

        return jsonify({'prediction': prediction_result})

    except ValueError as ve:
        # 处理数组转换或预测过程中可能出现的错误
         return jsonify({'error': f'处理特征时出错: {ve}'}), 400
    except Exception as e:
        # 通用错误处理程序
        app.logger.error(f"预测错误: {e}") # 记录错误以便调试
        return jsonify({'error': '预测过程中发生内部错误。'}), 500

# 4. 运行 Flask 应用程序
# 此代码块允许直接使用 `python app.py` 运行应用程序
# 在生产环境中，调试模式应为 False
if __name__ == '__main__':
    # 确保如果程序在 Docker 中运行，服务器可以从外部访问
    # 0.0.0.0 使其监听所有可用的网络接口
    app.run(host='0.0.0.0', port=5000, debug=False) 
```

**说明：**

1. **加载模型：** 我们使用 `joblib.load()` 加载预训练 (pre-training)模型文件。并包含了基本的错误处理。
2. **Flask 应用程序：** 我们初始化一个 Flask 应用程序。
3. **预测端点：**
   - 我们定义了一个 `/predict` 路由，它只接受 POST 请求（因为我们正在向它发送数据）。
   - 在 `predict` 函数内部，我们使用 `request.get_json()` 获取请求体中发送的 JSON 数据。
   - 我们期望 JSON 中有一个 `features` 字段，其中包含一个数值特征值列表（例如 `{"features": [5.1, 3.5, 1.4, 0.2]}`）。添加了基本的输入验证。
   - 特征被转换为 NumPy 数组并重塑，因为 scikit-learn 模型通常期望一个二维数组。
   - 使用特征调用 `model.predict()`。
   - 预测结果被提取并转换为标准 Python 类型，然后使用 `jsonify` 作为 JSON 响应返回。
   - 添加了错误处理以解决数据处理或预测期间的问题。
4. **运行应用程序：** `if __name__ == '__main__':` 代码块允许您使用 `python app.py` 在本地运行服务器进行测试。我们将 `host='0.0.0.0'` 设置为使其稍后可从容器外部访问，`port=5000` 是开发服务器的常见选择。`debug=False` 对于本地测试以外的任何情况都很重要。

### 步骤 2：定义依赖项

在 `requirements.txt` 中列出应用程序所需的 Python 包。

```text
# requirements.txt
Flask>=2.0.0,<3.0.0
scikit-learn>=1.0.0,<1.4.0 # 或与您的模型兼容的版本
joblib>=1.1.0
numpy>=1.21.0
```

*注意：根据您的环境或具体需求调整版本号。通常建议使用特定的兼容版本以获得稳定性。*

### 步骤 3：创建 Dockerfile

`Dockerfile` 为 Docker 提供指令，用于构建一个包含您的应用程序、其依赖项和必要运行环境的镜像。

编辑 `Dockerfile` 并添加以下内容：

```dockerfile
# Dockerfile

# 1. 使用官方 Python 运行时作为父镜像
# 使用 'slim' 版本可减小镜像大小
FROM python:3.9-slim

# 2. 在容器内设置工作目录
WORKDIR /app

# 3. 将 requirements 文件复制到容器的 /app 目录
COPY requirements.txt .

# 4. 安装 requirements.txt 中指定的任何所需软件包
# --no-cache-dir 减小层大小，--trusted-host 处理潜在的网络问题
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# 5. 将其余应用程序代码（app.py, model.joblib）复制到容器中
COPY . .

# 6. 暴露应用程序运行的端口
EXPOSE 5000

# 7. 定义容器启动时运行应用程序的命令
CMD ["python", "app.py"]
```

**说明：**

1. **`FROM python:3.9-slim`**：指定基础镜像。我们使用一个精简的 Python 3.9 镜像。
2. **`WORKDIR /app`**：设置容器内后续命令的默认目录。
3. **`COPY requirements.txt .`**：将 `requirements.txt` 文件从您的主机复制到容器的 `/app` 目录中。
4. **`RUN pip install ...`**：执行命令以安装 `requirements.txt` 中列出的依赖项。
5. **`COPY . .`**：将主机上当前目录中的所有剩余文件（包括 `app.py` 和 `model.joblib`）复制到容器的 `/app` 目录中。
6. **`EXPOSE 5000`**：通知 Docker 容器将在运行时监听端口 5000。这实际上并不会发布端口；它更像是文档。
7. **`CMD ["python", "app.py"]`**：指定基于此镜像的容器启动时要执行的默认命令。这将运行我们的 Flask 应用程序。

### 步骤 4：构建 Docker 镜像

现在，打开您的终端或命令提示符，导航到 `model_api_project` 目录，然后运行 Docker 构建命令：

```bash
docker build -t model-api:latest .
```

- `docker build`：用于构建镜像的命令。
- `-t model-api:latest`：用名称（`model-api`）和标签（`latest`）标记 (token)镜像。这使得以后更容易引用。
- `．`：指定构建上下文 (context)（当前目录），其中包含 `Dockerfile` 和所有必要文件。

Docker 将执行 `Dockerfile` 中的步骤，下载基础镜像，安装依赖项，并复制您的文件。首次运行时可能需要几分钟。

### 步骤 5：运行 Docker 容器

镜像成功构建后，您可以从中运行容器：

```bash
docker run -p 5001:5000 model-api:latest
```

- `docker run`：用于从镜像创建并启动容器的命令。
- `-p 5001:5000`：将容器的端口 5000 发布到您主机上的端口 5001。这意味着发送到您计算机上 `localhost:5001` 的请求将被转发到容器内部 Flask 应用程序监听的端口 5000。如果需要，您可以将 `5001` 更改为另一个可用端口。
- `model-api:latest`：指定用于容器的镜像。

您应该会看到指示 Flask 服务器正在运行的输出，类似于您在本地运行 `python app.py` 时的情况，如果找到了 `model.joblib`，还会包含“模型加载成功”的消息。

### 步骤 6：测试 API

容器运行后，打开另一个终端或使用 Postman 或 Insomnia 等工具向 API 发送 POST 请求。使用 `curl`：

```bash
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**说明：**

- `curl`：一个用于通过 URL 传输数据的命令行工具。
- `-X POST`：将 HTTP 方法指定为 POST。
- `http://localhost:5001/predict`：您正在运行的 API 端点的 URL（使用您发布的宿主端口）。
- `-H "Content-Type: application/json"`：设置内容类型标头，表示我们正在发送 JSON 数据。
- `-d '{"features": [5.1, 3.5, 1.4, 0.2]}'`：请求体中发送的数据负载。**请确保特征的数量和顺序与您的 `model.joblib` 所期望的匹配。**

如果成功，容器内运行的 API 将处理请求并返回一个 JSON 响应，例如：

```json
{"prediction": 0}
```

（实际的预测值 `0` 取决于您的特定 `model.joblib`）。

您可以通过返回到执行 `docker run` 的终端并按下 `Ctrl+C` 来停止正在运行的容器。

### 总结

恭喜！您已成功完成以下任务：

1. 使用 Flask 创建了一个简单的 Web API，用于从已保存的机器学习 (machine learning)模型提供预测。
2. 定义了项目的依赖项。
3. 编写了 `Dockerfile` 来指定容器构建过程。
4. 构建了一个包含您的应用程序及其环境的 Docker 镜像。
5. 从镜像运行了一个 Docker 容器，使您的 API 可被访问。
6. 通过发送预测请求测试了容器化的 API。

这构成了模型部署的开端。在此基础上，您可以考虑添加更多错误处理、输入验证、日志记录、部署到云平台，或者使用更多服务框架。本实践练习是在开发模型与使其在实际应用程序中可用之间架起桥梁的重要一步。

## 参考资料

- [Flask Documentation](https://flask.palletsprojects.com/en/) — Pallets (2023)
  Flask Web框架的官方指南，涵盖API开发、路由以及HTTP请求和响应的处理。
- [Docker Documentation](https://docs.docker.com/) — Docker Inc. (2023)
  Docker的官方资源，包含Dockerfile语法、镜像构建和容器运行时管理。
- [Designing Machine Learning Systems: New Ways of Working with AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Huyen, Chip (2022)
  Publisher: O'Reilly Media
  关于构建、部署和维护稳健机器学习系统的指南，为模型部署实践提供背景。

---

[上一节](05-%E6%A8%A1%E5%9E%8B%E7%9B%91%E6%8E%A7%E5%9F%BA%E6%9C%AC%E8%A6%81%E7%82%B9.md)
