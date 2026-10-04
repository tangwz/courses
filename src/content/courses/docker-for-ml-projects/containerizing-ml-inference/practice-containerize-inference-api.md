---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-inference"
lesson: "practice-containerize-inference-api"
sourceId: 5329
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/practice-containerize-inference-api"
title: "实操：容器化简单推理API"
description: "将预训练模型和简单的Flask/FastAPI应用程序打包到优化的Docker容器中。"
order: 7
plots: []
sourceHash: "92e0d43d1a08d8f9b2487cbd8d859d9dc89a43c3c0b8e077b5d058e3a9c55e71"
sourceCorrections: []
---

我们将使用一个预训练 (pre-training)模型，将其封装在基于Flask的轻量级Web应用程序中，然后创建一个Dockerfile，将所有内容打包到一个高效、可运行的容器里。

**先决条件：**

- 一个保存到文件的预训练机器学习 (machine learning)模型。本例中，我们假设您有一个保存为`model.pkl`的scikit-learn模型。您可以训练一个简单模型（例如，在Iris数据集上进行逻辑回归）并使用`joblib`或`pickle`保存它。
- 对Python和Flask有基本了解。
- 您的系统已安装并运行Docker。

**项目结构：**

请按以下方式组织您的项目文件：

```
.
├── app.py             # Flask应用程序代码
├── model.pkl          # 您的预训练模型文件
├── requirements.txt   # Python依赖项
└── Dockerfile         # 构建Docker镜像的说明
```

**1. 创建推理 (inference)API (`app.py`)**

这个Python脚本将使用Flask来创建一个Web服务器。它会加载我们的预训练模型，并公开一个端点（例如`/predict`），该端点接受数据，使用模型进行预测，并返回结果。

```python
# app.py
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# 初始化Flask应用
app = Flask(__name__)

# 加载预训练模型
# 确保model.pkl在同一目录中或提供正确路径
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 从POST请求获取数据
        data = request.get_json(force=True)

        # 确保数据是预期格式（例如，特征列表）
        # 根据您的模型输入要求调整此部分
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON payload'}), 400

        features = np.array(data['features'])

        # 执行预测
        # 如果您的模型需要一个二维数组（例如，单个样本），则进行形状调整
        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = model.predict(features)
        prediction_proba = None
        if hasattr(model, "predict_proba"):
             # 如果模型支持，获取概率
             prediction_proba = model.predict_proba(features).tolist()

        # 将预测结果作为JSON响应返回
        response = {
            'prediction': prediction.tolist(),
        }
        if prediction_proba:
            response['probabilities'] = prediction_proba

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # 简单健康检查端点
    # 您可以在这里添加检查（例如，模型是否加载）
    if model is not None:
        return jsonify({'status': 'ok'}), 200
    else:
        return jsonify({'status': 'error', 'reason': 'Model not loaded'}), 500

if __name__ == '__main__':
    # 使用Flask内置服务器运行应用程序，用于开发
    # 生产环境请使用Gunicorn等WSGI服务器（在Dockerfile中配置）
    app.run(host='0.0.0.0', port=5000, debug=False) # 将debug=False设置为模拟生产环境
```

此脚本定义了两个路由：

- `/predict`：接受包含`features`键的JSON数据的POST请求。它使用已加载的`model.pkl`进行预测并返回结果。
- `/health`：一个简单的GET端点，用于检查服务是否正在运行以及模型是否已加载。

**2. 定义依赖项 (`requirements.txt`)**

列出您的API所需的Python库。生产环境，我们包含`gunicorn`，一个WSGI服务器。

```
# requirements.txt
flask
scikit-learn   # 或您的模型使用的库（例如，tensorflow, torch）
numpy
joblib         # 如果您使用pickle保存模型，则为pickle
gunicorn       # 生产环境的WSGI服务器
```

**3. 创建`Dockerfile`**

此文件包含Docker用于构建镜像的指令。我们将从一个基础版本开始，然后展示一个优化的多阶段构建。

**基础版Dockerfile：**

```dockerfile
# Dockerfile (基础版)

# 1. 基础镜像：使用官方Python运行时
FROM python:3.9-slim

# 2. 设置工作目录：定义后续指令的上下文
WORKDIR /app

# 3. 复制依赖文件：首先复制requirements文件以便层缓存
COPY requirements.txt .

# 4. 安装依赖项：安装Python包
#    --no-cache-dir: 通过不存储pip缓存来减小镜像大小
#    --upgrade pip: 确保pip是最新版本
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 复制应用程序代码和模型：复制您的应用程序的其余部分
COPY . .

# 6. 暴露端口：通知Docker容器监听端口5000
EXPOSE 5000

# 7. 定义运行命令：指定使用Gunicorn运行应用程序的命令
#    --bind 0.0.0.0:5000: 监听容器内部所有网络接口
#    app:app: 指的是'app.py'文件中名为'app'的Flask应用对象
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**说明：**

- `FROM python:3.9-slim`：使用轻量级的官方Python镜像。
- `WORKDIR /app`：将容器内的当前目录设置为`/app`。
- `COPY requirements.txt .`：首先只复制`requirements`文件。Docker会缓存层，因此如果`requirements.txt`没有改变，`pip install`层就不需要重新构建，这会加速后续构建。
- `RUN pip install...`：使用`pip`安装依赖项。`--no-cache-dir`阻止pip存储下载的包，节省空间。
- `COPY . .`：将项目其余文件（`app.py`、`model.pkl`）复制到容器中的`/app`目录。
- `EXPOSE 5000`：记录容器内的应用程序将监听端口5000。这不会发布端口；它只是元数据。
- `CMD ["gunicorn", ...]`：指定容器启动时运行的默认命令。我们使用Gunicorn作为生产就绪的服务器，将其绑定到`0.0.0.0`，使其可以从容器外部访问（当映射时），并指定我们的Flask应用程序对象（`app.py`中的`app`）。

**4. 构建Docker镜像**

在项目目录（`Dockerfile`所在位置）中打开终端并运行构建命令：

```bash
docker build -t ml-inference-api:basic .
```

- `docker build`：构建镜像的命令。
- `-t ml-inference-api:basic`：使用名称（`ml-inference-api`）和标签（`basic`）标记 (token)镜像。
- `.`：指定构建上下文 (context)（当前目录）。Docker将此目录中的文件发送到Docker守护进程。

**5. 运行容器**

现在，从您刚刚构建的镜像运行一个容器：

```bash
docker run -d -p 5001:5000 --name my_api ml-inference-api:basic
```

- `docker run`：创建并启动容器的命令。
- `-d`：以分离模式（后台）运行容器。
- `-p 5001:5000`：将主机上的端口5001映射到容器内的端口5000。这允许您通过`localhost:5001`访问API。
- `--name my_api`：为正在运行的容器指定一个名称，以便于管理。
- `ml-inference-api:basic`：要运行的镜像。

您可以使用`docker ps`检查容器是否正在运行。

**6. 测试API**

您可以使用`curl`或一个简单Python脚本来测试正在运行的API。假设您的模型需要4个特征：

从终端使用`curl`：

```bash
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}' # Iris特征示例

# 预期输出（会根据您的模型而异）：
# {"prediction":[0]} or {"prediction":[0], "probabilities":[[0.98, 0.01, 0.01]]}
```

检查健康端点：

```bash
curl http://localhost:5001/health
# 预期输出：
# {"status":"ok"}
```

**7. 使用多阶段构建进行优化**

基础镜像包含构建工具和一些可能大型但非运行推理服务必需的库。多阶段构建可以创建更小、更安全的生产镜像。

**多阶段Dockerfile：**

```dockerfile
# Dockerfile (多阶段)

# ---- 构建阶段 ----
# 使用完整Python镜像安装依赖项，包括任何构建时所需
FROM python:3.9 as builder

WORKDIR /build

# 如果需要，安装构建必需品（例如，用于C扩展）
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 将依赖项安装到指定目录
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix="/install" -r requirements.txt

# ---- 运行阶段 ----
# 为最终运行时环境使用一个最小基础镜像
FROM python:3.9-slim

WORKDIR /app

# 只从构建阶段复制已安装的包
COPY --from=builder /install /usr/local

# 复制应用程序代码和模型
COPY app.py .
COPY model.pkl .

# 暴露端口
EXPOSE 5000

# 定义运行命令（与之前相同）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**变更说明：**

1. **`FROM python:3.9 as builder`**：定义了名为`builder`的第一个阶段。如果编译需要，此阶段可以使用更大的镜像。
2. **`pip install --prefix="/install"`**：将包安装到`builder`阶段内的特定目录（`/install`），而不是默认的系统Python位置。
3. **`FROM python:3.9-slim`**：为最终运行时阶段启动一个*新的*、干净的、最小的基础镜像。
4. **`COPY --from=builder /install /usr/local`**：这是重要部分。它*只*将`/install`目录（我们的依赖项安装的位置）的内容从`builder`阶段复制到最终镜像的标准库位置（`/usr/local`）。`builder`阶段的构建工具和缓存会被丢弃。
5. **`COPY app.py .`，`COPY model.pkl .`**：复制必要的应用程序文件。

**构建与比较：**

构建优化后的镜像：

```bash
docker build -t ml-inference-api:optimized .
```

比较镜像大小：

```bash
docker images ml-inference-api
```

您会发现`ml-inference-api:optimized`比`ml-inference-api:basic`小很多。像运行基础版本一样运行它（如果需要，调整端口）：

```bash
# 如果正在运行，请先停止并删除之前的容器
docker stop my_api
docker rm my_api

# 运行优化版本
docker run -d -p 5001:5000 --name my_api_optimized ml-inference-api:optimized
```

再次使用`curl`进行测试，就像之前一样。

**总结：**

在此实践练习中，您成功完成了以下事项：

1. 创建了一个简单的Flask API，用于提供预训练模型的预测服务。
2. 在`requirements.txt`中定义了Python依赖项。
3. 编写了`Dockerfile`，用于打包API、模型和依赖项。
4. 构建了一个包含推理服务的基础Docker镜像。
5. 实现了多阶段构建，以创建更小、更优化的生产镜像。
6. 运行了容器化的API并测试了其功能。

您现在掌握了一个将训练好的模型转变为可分发、容器化推理服务的实用流程，为后续部署步骤做好了准备。请记住根据您的具体模型及其依赖项来调整`app.py`逻辑和`requirements.txt`。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker, Inc. (2024)
  Docker的官方文档，提供了关于基本概念、Dockerfile语法、镜像构建、容器管理和多阶段构建的信息。
