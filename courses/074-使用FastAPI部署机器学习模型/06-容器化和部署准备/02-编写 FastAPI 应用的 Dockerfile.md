# 编写 FastAPI 应用的 Dockerfile

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/writing-dockerfile-fastapi)

[返回章节目录](README.md) · [返回课程目录](../README.md)

容器化旨在为应用创建一致的环境。在这一过程中，`Dockerfile` 是定义应用*如何*打包的核心组件。你可以将 `Dockerfile` 视为一个食谱或一系列指令，Docker 遵循这些指令来构建一个镜像，其中包含你的 FastAPI 应用、其依赖项、所需数据文件（比如你的机器学习 (machine learning)模型）以及运行它所需的信息。

`Dockerfile` 只是一个简单的文本文件，文件名为 `Dockerfile`（无扩展名），放置在项目的根目录中。在这个文件中，你将一行一行地写入一系列命令，这些命令指定了组装镜像的步骤。

### Dockerfile 的基本指令

虽然你可以在 `Dockerfile` 中使用许多可能的指令，但 FastAPI 应用的典型设置通常涉及以下几种常用指令：

- `FROM`：指定用于构建的基础镜像。每个 `Dockerfile` 都必须以 `FROM` 指令开头。对于 Python 应用，通常会使用官方的 Python 镜像（例如 `python:3.9`）。这提供了一个预装有 Python 和 pip 的初始 Linux 环境。
- `WORKDIR`：设置后续指令（如 `RUN`、`CMD`、`COPY` 和 `ADD`）的工作目录。如果目录不存在，`WORKDIR` 会创建它。它有助于组织容器的文件系统。
- `COPY`：将文件或目录从本地机器（构建上下文 (context)）复制到容器镜像的文件系统中。你会使用它将应用代码、`requirements.txt` 文件和机器学习 (machine learning)模型工件复制到镜像中。
- `RUN`：在当前镜像之上创建一个新层并执行命令。这常用于使用 `pip install` 安装 Python 依赖项。每条 `RUN` 指令都会创建一个新的镜像层，这会影响缓存和镜像大小。
- `EXPOSE`：通知 Docker 容器在运行时监听指定的网络端口。这实际上并不会发布端口；它起到构建镜像者和运行容器者之间的文档作用。对于使用 Uvicorn 默认设置运行的 FastAPI 应用，这通常是端口 8000。
- `CMD`：提供从镜像启动容器时执行的默认命令。一个 `Dockerfile` 中只能有一条 `CMD` 指令。如果你指定可执行文件，它应作为第一个参数 (parameter)。对于运行 FastAPI，此命令通常会启动 Uvicorn 服务器。

### FastAPI 机器学习 (machine learning)应用的 Dockerfile 示例

让我们把这些指令组合起来，形成一个实用的 `Dockerfile`，用于我们正在构建的机器学习预测服务。假设你的项目结构如下所示：

```
.
├── app/
│   ├── main.py
│   ├── models/
│   │   └── your_model.joblib
│   └── ... (other python files)
├── requirements.txt
└── Dockerfile
```

这是一个 `Dockerfile` 示例：

```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置容器中的工作目录
WORKDIR /app

# 将 requirements 文件复制到容器的 /app 目录
COPY requirements.txt .

# 安装 requirements.txt 中指定的所需包
# 使用 --no-cache-dir 减少镜像大小
# 使用 --compile 对 Python 文件进行字节编译，以加快启动速度（可选）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --compile -r requirements.txt

# 将应用代码和模型文件复制到容器的 /app 目录
COPY ./app /app/app
COPY ./app/models /app/app/models

# 使端口 8000 可从此容器外部访问
EXPOSE 8000

# 定义环境变量（可选，也可在运行时设置）
ENV MODEL_PATH=/app/app/models/your_model.joblib

# 容器启动时使用 Uvicorn 运行 main.py
# 监听 0.0.0.0 以便从容器外部访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 理解示例

我们来分步解析这个 `Dockerfile` 示例：

1. `FROM python:3.9-slim`：我们从官方 Python 3.9 镜像的 slim 版本开始。使用 `slim` 变体通常会使镜像大小比完整镜像小，因为它们只包含运行 Python 所需的最少包。
2. `WORKDIR /app`：我们将容器内的默认目录设置为 `/app`。所有后续命令都将相对于此路径运行。
3. `COPY requirements.txt .`：我们将项目目录中的 `requirements.txt` 文件*只*复制到容器内的 `/app` 目录中。注意 `.` 表示目标目录（由于 `WORKDIR`，此处为 `/app`）。
4. `RUN pip install ...`：这是一个重要步骤。我们首先升级 `pip` 本身，然后安装 `requirements.txt` 中列出的所有库。
   - `--no-cache-dir`：此标志告诉 pip 不要存储下载缓存，这有助于减小镜像层的大小。
   - `--compile`：这会预编译 Python 源文件为字节码（`.pyc`），可以稍微加快应用启动速度。
   - 我们先复制 `requirements.txt` 并运行 `pip install`，*再*复制其余应用代码。这利用了 Docker 的构建缓存。如果 `requirements.txt` 没有变化，Docker 可以重用此 `RUN` 指令创建的层，即使应用代码发生变化，也能加快后续构建速度。
5. `COPY ./app /app/app`：我们将本地 `app` 目录（包含 `main.py` 等）的内容复制到容器 `/app` 目录内名为 `app` 的子目录中。目标路径变为 `/app/app`。
6. `COPY ./app/models /app/app/models`：同样，我们将机器学习 (machine learning)模型工件从本地 `app/models` 目录复制到容器内的 `/app/app/models` 目录中。你可以根据模型相对于 `Dockerfile` 的存储位置进行调整。
7. `EXPOSE 8000`：我们记录容器内的应用将监听端口 8000。
8. `ENV MODEL_PATH=...`：这会在容器内设置一个环境变量。你的 FastAPI 应用可以读取此环境变量以了解从何处加载模型文件，从而使路径可配置。
9. `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`：这定义了从该镜像启动容器时将运行的命令。它执行 `uvicorn` 服务器，指示其运行 `app/main.py` 模块中找到的 `app` 对象。
   - `--host 0.0.0.0`：这很重要。它使 Uvicorn 监听容器内所有可用的网络接口，从而允许来自容器外部的连接（当端口发布时）。使用默认的 `127.0.0.1` 只允许来自容器内部的连接。
   - `--port 8000`：与 `EXPOSE` 指令中指定的端口匹配。

### `.dockerignore` 文件

类似于 `.gitignore`，你可以在与 `Dockerfile` 相同的目录中创建一个 `.dockerignore` 文件。此文件列出了应从发送到 Docker 守护进程的构建上下文 (context)中排除的文件和目录模式。这可以避免不必要的过大构建上下文，并防止将敏感信息或不相关文件（如虚拟环境、`.pyc` 文件、构建工件、`.git` 目录）复制到镜像中。

一个典型的 `.dockerignore` 文件可能如下所示：

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git/
.pytest_cache/
```

通过定义这个 `Dockerfile`，你提供了一套清晰、可重复的指令，用于打包你的 FastAPI 机器学习 (machine learning)应用。下一步是使用此文件来构建 Docker 镜像并将其作为容器运行。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) — Docker (2024)
  Dockerfile语法、命令和参数的官方详细指南，提供了本节中讨论的每个指令的具体信息。
- [Deployment and production](https://fastapi.tiangolo.com/deployment/) — Sebastián Ramírez (2024)
  FastAPI官方文档的这部分内容提供了部署FastAPI应用程序的指导，包括针对Uvicorn生产使用和容器化策略的建议。
- [Deployment](https://www.uvicorn.org/deployment/) — Encode OSS (2024)
  Publisher: Encode OSS
  Uvicorn官方文档提供了部署ASGI服务器的说明和最佳实践，这对于在生产环境（包括Docker容器内）运行FastAPI应用程序至关重要。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker (2024)
  Publisher: Docker
  这份Docker官方指南提供了创建高效、可维护和安全的Docker镜像的建议，涵盖了优化层、缓存和减小镜像大小等主题，直接适用于构建基于Python的应用程序。

---

[上一节](01-Docker%20%E5%BA%94%E7%94%A8%E6%89%93%E5%8C%85%E4%BB%8B%E7%BB%8D.md) · [下一节](03-%E5%9C%A8%20Docker%20%E9%95%9C%E5%83%8F%E4%B8%AD%E6%89%93%E5%8C%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B.md)
