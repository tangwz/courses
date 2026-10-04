---
course: "introduction-to-mlops"
chapter: "model-deployment-and-serving"
lesson: "practice-deploy-model-docker"
sourceId: 7261
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving/practice-deploy-model-docker"
title: "动手实践：将模型打包并部署为 Docker 容器"
description: "将 Python 机器学习模型及其依赖项打包到 Docker 容器中进行部署的实用指南。"
order: 7
plots: []
sourceHash: "5e17ac705b95715fff7d87071e91abba021e4317ea5beee2cdca95aa420d71ae"
sourceCorrections: []
---

将机器学习 (machine learning)模型打包成独立、可移植的 Docker 容器是现代模型部署的基本技能。本指南演示了如何结合 Docker 容器化技术，并将模型包装在 Web API 中来实现这一目标。

我们的目标是将预训练 (pre-training)模型、提供服务的 Web 应用程序及其所有依赖项捆绑到一个称为 Docker 镜像的单元中。通过这个镜像，我们可以启动一个容器，将模型作为一个隔离的服务运行。

### 前提条件

在开始之前，请确保您的计算机上已安装并运行了 Docker Desktop。您可以从 Docker 官方网站下载。

我们将为本次练习创建几个文件。让我们为项目建立一个新目录，以保持条理性。

> 简单模型部署项目的文件结构。

### 第 1 步：创建应用程序和模型文件

首先，我们需要准备打包的内容。我们将使用一个简单的 `scikit-learn` 模型和一个 Flask 应用程序来提供服务。您不需要亲自训练模型；在本次练习中，我们假设您已经有一个 `model.pkl` 文件。

**模型文件 (`model.pkl`)**

在本指南中，`model.pkl` 代表一个简单的预训练 (pre-training)回归模型。它接收四个数值特征作为输入，并返回一个预测结果。

**应用程序 (`app.py`)**

这个 Python 脚本使用 Flask 框架创建一个简单的 Web 服务器。它加载 `model.pkl` 文件并提供一个 `/predict` 端点，该端点接收包含 JSON 数据的 POST 请求。

创建一个名为 `app.py` 的文件并添加以下代码：

```python
import joblib
from flask import Flask, request, jsonify

# 1. 初始化 Flask 应用
app = Flask(__name__)

# 2. 加载预训练模型
# 仅在应用启动时加载一次
model = joblib.load('model.pkl')

# 3. 定义预测端点
@app.route('/predict', methods=['POST'])
def predict():
    """
    接收包含 JSON 数据的 POST 请求，
    运行模型并返回预测结果。
    """
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()

        # 'features' 键应包含一个数字列表
        features = data['features']

        # 运行预测
        prediction = model.predict([features])

        # 以 JSON 格式返回结果
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # 处理错误，例如输入格式错误
        return jsonify({'error': str(e)}), 400

# 4. 启动 Flask 服务器
if __name__ == '__main__':
    # 服务器可以通过 http://0.0.0.0:5000 访问
    app.run(host='0.0.0.0', port=5000)
```

**依赖项 (`requirements.txt`)**

我们的应用程序依赖于 `scikit-learn`（用于模型）和 `Flask`（用于 Web 服务器）。我们必须在 `requirements.txt` 文件中列出这些依赖项，以便 Docker 知道需要安装什么。

创建一个名为 `requirements.txt` 的文件：

```
scikit-learn==1.0.2
Flask==2.1.2
joblib==1.1.0
```

> 注意：使用特定版本是确保可复现性的良好做法。您的应用程序将始终使用这些确切的库版本构建，从而避免库更新带来的意外情况。

### 第 2 步：编写 Dockerfile

`Dockerfile` 是一个文本文件，包含构建 Docker 镜像的一系列指令。它就像是创建应用程序环境的食谱。

在项目目录中创建一个名为 `Dockerfile`（无扩展名）的文件，并添加以下指令。我们将在下方说明每一行的作用。

```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置容器内的当前目录
WORKDIR /app

# 先复制需求文件以利用 Docker 缓存
COPY requirements.txt .

# 安装 requirements.txt 中指定的任何所需软件包
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序的其余代码
COPY . .

# 暴露应用运行的端口
EXPOSE 5000

# 定义运行应用程序的命令
CMD ["python", "app.py"]
```

让我们分解每条指令的功能：

- `FROM python:3.9-slim`：每个 Docker 镜像都从基础镜像开始。这里我们使用官方 Python 3.9 镜像。`-slim` 版本体积较小，可以得到更轻量的最终镜像。
- `WORKDIR /app`：这会将容器内部的工作目录设置为 `/app`。后续的所有命令（`COPY`、`RUN`、`CMD`）都将在此目录下执行。
- `COPY requirements.txt .`：我们将 `requirements.txt` 文件复制到容器的工作目录中。我们在复制其余代码 *之前* 执行此操作，是为了利用 Docker 的分层缓存。如果 Python 代码发生了变化但依赖项没变，Docker 可以复用未改变的层，从而使后续构建速度快得多。
- `RUN pip install --no-cache-dir -r requirements.txt`：此命令在镜像内部执行 `pip install`，安装 `requirements.txt` 文件中列出的所有库。`--no-cache-dir` 标志通过不存储下载缓存来减小镜像体积。
- `COPY . .`：这将本地项目目录（构建上下文 (context)）中的所有剩余文件复制到容器的工作目录（`/app`）中。这包括 `app.py` 和 `model.pkl`。
- `EXPOSE 5000`：此指令通知 Docker 容器在运行时监听指定的网络端口。它主要用于文档说明，实际上并不会发布端口。
- `CMD ["python", "app.py"]`：这指定了从该镜像启动容器时运行的默认命令。在我们的例子中，它通过运行 `python app.py` 来启动 Flask Web 服务器。

> 使用 `Dockerfile` 构建静态“镜像”，然后使用该镜像启动一个或多个运行中“容器”的过程。

### 第 3 步：构建 Docker 镜像

既然有了 `Dockerfile`，我们就可以用它来构建镜像。打开终端或命令提示符，进入项目目录，运行以下命令：

```bash
docker build -t my-ml-app .
```

让我们分析一下这个命令：

- `docker build`：从 Dockerfile 构建镜像的命令。
- `-t my-ml-app`：`-t` 标志代表“标签（tag）”。它为我们的镜像命名，方便以后引用。我们将其命名为 `my-ml-app`。
- `.`：最后这个点告诉 Docker 在当前目录中查找 `Dockerfile`。该目录也是“构建上下文 (context)”，这里的所有文件都会发送给 Docker 守护进程用于构建过程。

Docker 现在将逐步执行 `Dockerfile` 中的指令。您将看到每一层构建的输出。

### 第 4 步：运行 Docker 容器

构建完成后，您的本地计算机上就有了一个名为 `my-ml-app` 的 Docker 镜像。您可以通过运行 `docker images` 查看它。

现在，让我们从这个镜像运行一个容器：

```bash
docker run -p 5001:5000 my-ml-app
```

该命令的作用如下：

- `docker run`：启动新容器的命令。
- `-p 5001:5000`：这是端口映射标志。它将本地计算机（宿主机）上的端口 `5001` 映射到容器内部的端口 `5000`。我们的 Flask 应用在容器内监听 `5000` 端口，通过此映射，可以从宿主机的 `5001` 端口访问它。
- `my-ml-app`：用于创建容器的镜像名称。

您应该会看到来自 Flask 服务器的输出，表明它已启动并正在监听 `http://0.0.0.0:5000`。

### 第 5 步：测试运行中的服务

您的模型现在运行在一个隔离的容器中，并且可以在本地访问。让我们向它发送一个预测请求。

打开一个*新*终端窗口（保持运行容器的窗口开启），使用 `curl` 之类的工具发送 POST 请求：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.5, 3.5, 4.5]}' \
  http://localhost:5001/predict
```

如果一切运行正常，您将收到来自模型的 JSON 响应，如下所示：

```json
{"prediction":15.5}
```

> 注意：确切的预测值取决于您使用的 `model.pkl` 文件。其重点在于收到了有效的 JSON 响应。

祝贺您！您已成功将机器学习 (machine learning)模型和 Web 应用程序打包到 Docker 容器中并提供了预测服务。这个容器现在是一个可移植的产物。您可以将其推送到容器镜像仓库，并在任何安装了 Docker 的机器上运行，且能确保环境和依赖项完全一致。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) — Docker (2023)
  官方文档，详细介绍了Dockerfile指令和构建Docker镜像的最佳实践，对于应用程序容器化至关重要。
- [Flask Documentation](https://flask.palletsprojects.com/en/latest/) — Pallets (2023)
  Flask Web框架的官方指南，对于创建基于Python的Web服务和REST API以提供机器学习模型至关重要。
- [Introducing MLOps: How to go from model to production](https://books.google.com/books/about/Introducing_MLOps.html?id=z1_zDwAAQBAJ) — Mark Treveil, Nicolas Omont, Aurélien Simon, Andreas Schuderer, Hannes Hapke, Noah Gift, and David Noël (2020)
  Publisher: O'Reilly Media
  一本全面的书籍，涵盖了MLOps的原则和实践，包括使用容器化部署机器学习模型的策略。
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html) — scikit-learn developers (2023)
  官方文档，解释了如何使用joblib或pickle保存和加载scikit-learn模型，这是模型部署的基础。
