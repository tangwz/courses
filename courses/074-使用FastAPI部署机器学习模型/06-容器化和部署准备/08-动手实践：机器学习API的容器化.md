# 动手实践：机器学习API的容器化

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/practice-containerizing-ml-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

这项实践操作通过应用Docker和容器化原理来封装机器学习 (machine learning)API。它假设你有一个功能完善的FastAPI应用，包括你训练好的模型文件（例如，`.joblib` 或 `.pkl` 文件），并且你的系统上已经安装并运行了Docker。

### 项目结构概述

在创建 `Dockerfile` 之前，让我们回顾一下典型的项目布局。你的结构可能如下所示：

```
my_ml_api/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app instance, imports routers
│   ├── api/           # Routers define endpoints
│   │   ├── __init__.py
│   │   └── predict.py # Contains the /predict endpoint
│   ├── models/        # Pydantic models for request/response
│   │   └── __init__.py
│   └── core/          # Model loading and inference logic
│       ├── __init__.py
│       └── inference.py
├── models/            # Directory for ML model artifacts
│   └── sentiment_model.joblib # Example model file
├── tests/             # Application tests
│   └── ...
├── Dockerfile         # We will create this file
└── requirements.txt   # Python dependencies
```

这是说明这种结构的图表：

> 项目结构展示了应用代码（`app/`）、模型文件（`models/`）、测试（`tests/`）和配置文件（`Dockerfile`、`requirements.txt`）。

确保你的机器学习 (machine learning)模型文件（例如，`sentiment_model.joblib`）位于项目根目录下的 `models/` 文件夹中，并且你的 `requirements.txt` 文件准确列出了所有必需的包（如 `fastapi`、`uvicorn`、`scikit-learn`、`joblib`、`pydantic`）。

### 创建Dockerfile

在终端中进入你项目的根目录（`my_ml_api/`）。创建一个名为 `Dockerfile`（无扩展名）的新文件，并在文本编辑器中打开它。

将以下内容逐步添加到你的 `Dockerfile` 中：

```dockerfile
# 1. 基础镜像：使用官方Python运行时作为父镜像
# 我们使用特定版本以保证可复现性，并使用“slim”变体来减小镜像大小。
FROM python:3.9-slim

# 2. 设置工作目录：定义容器内的工作目录
WORKDIR /app

# 3. 复制依赖文件：首先复制requirements文件以便Docker缓存
COPY requirements.txt requirements.txt

# 4. 安装依赖：安装requirements.txt中指定的Python依赖
# --no-cache-dir 减少镜像大小，--upgrade pip 确保我们有最新版本的pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 复制应用代码和模型：复制其余的应用代码和机器学习模型
COPY ./app /app/app
COPY ./models /app/models

# 6. 暴露端口：通知Docker容器在运行时监听端口8000
EXPOSE 8000

# 7. 运行命令：指定使用Uvicorn运行应用的命令
# 我们使用 0.0.0.0 使服务器可从容器外部访问。
# 端口8000与EXPOSE指令一致。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

让我们逐一解释这些指令：

- `FROM python:3.9-slim`：指定基础镜像。使用特定版本标签（`3.9-slim`）可确保一致性，而 `slim` 变体有助于减小镜像大小。
- `WORKDIR /app`：设置容器内后续命令的默认目录。
- `COPY requirements.txt requirements.txt`：首先复制 `requirements` 文件。Docker 会缓存各层；如果 `requirements.txt` 文件没有变化，那么在后续构建中，pip 安装层可以从缓存中直接使用，从而加快构建速度。
- `RUN pip install ...`：执行依赖项的安装。使用 `&& \` 链接命令，`--no-cache-dir` 可以阻止 pip 存储缓存，从而减小最终镜像大小。
- `COPY ./app /app/app` 和 `COPY ./models /app/models`：将你的应用源代码（`app` 目录）和机器学习 (machine learning)模型（`models` 目录）复制到容器工作目录（`/app`）内的指定位置。
- `EXPOSE 8000`：表明容器内的应用将监听端口8000。这实际上并不会发布端口；它更多是作为信息，可以被自动化系统使用。
- `CMD ["uvicorn", ...]`：定义了当容器从此镜像启动时要执行的默认命令。它会启动Uvicorn服务器，指示其运行在 `app/main.py` 中找到的FastAPI应用实例（`app`），监听所有可用的网络接口（`0.0.0.0`），并使用端口8000。

### 构建Docker镜像

在你的项目根目录中创建 `Dockerfile` 后，你现在可以构建Docker镜像了。打开终端，进入项目根目录（`my_ml_api/`），然后运行以下命令：

```bash
docker build -t my-ml-api:latest .
```

让我们分析一下这个命令：

- `docker build`：用于从 `Dockerfile` 构建镜像的命令。
- `-t my-ml-api:latest`：`-t` 标志用于给镜像打标签。我们将其命名为 `my-ml-api` 并赋予 `latest` 标签。你可以使用其他标签，如版本号（例如，`my-ml-api:0.1.0`）。打标签使得管理和引用镜像变得更简单。
- `.`：这表示构建上下文 (context)，即当前目录。Docker会将此目录中的文件和文件夹（如果存在 `.dockerignore` 则会遵循其规则）发送给Docker守护进程，以便在构建过程中使用。

你会看到Docker执行 `Dockerfile` 中定义的每个步骤。首次执行时可能需要几分钟，尤其是在依赖项安装步骤。如果依赖项没有改变，后续构建会更快，这得益于Docker的层缓存。

完成后，你可以通过列出本地Docker镜像来验证镜像是否已创建：

```bash
docker images
```

你应该在列表中看到带有 `latest` 标签的 `my-ml-api`。

### 运行Docker容器

现在你有了镜像，可以将其作为容器运行了：

```bash
docker run -d -p 8000:8000 --name ml-api-container my-ml-api:latest
```

标志说明：

- `docker run`：用于从镜像创建并启动容器的命令。
- `-d`：在分离模式（后台）运行容器，并打印容器ID。
- `-p 8000:8000`：将容器的端口发布到主机。它将你主机上的端口8000映射到容器内部的端口8000（Uvicorn正在监听的端口，由 `EXPOSE` 和 `CMD` 指定）。格式为 `主机端口:容器端口`。
- `--name ml-api-container`：为运行中的容器分配一个可识别的名称，使其更易于管理（例如，查看日志、停止）。
- `my-ml-api:latest`：指定要运行的镜像。

### 验证运行中的容器

你可以使用以下命令检查容器是否正在运行：

```bash
docker ps
```

此命令列出所有正在运行的容器。你应该看到 `ml-api-container` 被列出，显示它正在运行，并显示端口映射 `0.0.0.0:8000->8000/tcp`。

现在，测试你的API。打开你的网络浏览器或使用 `curl` 等工具向你的预测端点发送请求，该端点现在可通过你主机的8000端口访问：

```bash
# Example using curl, assuming your endpoint is /predict
# and accepts JSON input like {"text": "some input"}
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "FastAPI is great for ML models!"}'
```

将URL路径（`/predict`）和数据（`-d '...'`）替换为你的API端点和预期输入格式的具体内容。你应该收到模型返回的预测响应，该响应由Docker容器内运行的FastAPI应用提供。

要查看容器内应用生成的日志（对调试很有用），请使用 `docker logs` 命令，后跟容器名称：

```bash
docker logs ml-api-container
```

要停止容器：

```bash
docker stop ml-api-container
```

要移除容器（停止后）：

```bash
docker rm ml-api-container
```

你已成功将你的FastAPI机器学习 (machine learning)应用容器化！你现在拥有了一个包含应用代码、依赖项和机器学习模型的独立Docker镜像。这个镜像可以在不同环境中保持一致地运行，从而实现可靠的部署。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker Inc. (2024)
  提供关于 Dockerfile 创建、镜像构建、容器管理以及底层容器化原则的全面指南。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  学习 FastAPI 的官方资源，涵盖 API 开发、依赖管理和应用程序结构。
- [Uvicorn Documentation](https://www.uvicorn.org/) — Tom Christie (2024)
  Uvicorn 文档，Uvicorn 是运行 FastAPI 应用程序的 ASGI 服务器，详细说明其配置和使用。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff and Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本关于理解和使用 Docker 的实用指南，包含核心概念、Dockerfile 最佳实践和容器编排。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://books.google.com/books/about/Building_Machine_Learning_Powered_Applic.html?id=L8YXEAAAQBAJ) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  该书提供有关部署机器学习模型的指南，包括使用容器进行打包和服务，与课程的总体目标一致。

---

[上一节](07-%E7%94%9F%E4%BA%A7%E9%83%A8%E7%BD%B2%E5%87%86%E5%A4%87%20%28Gunicorn-Uvicorn%29.md)
