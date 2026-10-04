# 使用 Docker 将 LangChain 应用容器化

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production/containerizing-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将 LangChain 应用从开发环境迁移到生产环境，需要以一种无论在何处运行都能确保一致性和可靠性的方式进行打包。容器化，特别是使用 Docker，为此提供了一个强大的方法。它允许您将应用程序代码、其依赖项（如 Python 库、LangChain 本身、如果需要的话特定模型文件）以及系统配置捆绑成一个单一、可移植的单元，称为容器镜像。

想象一下将您的应用直接部署到不同的服务器上。您可能会遇到“在我的机器上能运行”的问题，原因在于操作系统、已安装的 Python 版本或系统库的微小差异。容器通过创建一个隔离环境来解决这个问题，使您的应用始终以相同的方式运行。

### 理解 Docker 的基本认识

Docker 的核心是使用**镜像**和**容器**。

- **镜像：** 一个只读模板，包含应用程序代码、运行时（如 Python）、库、环境变量和配置文件。您可以通过特殊文件 `Dockerfile` 中定义的指令来构建镜像。可以将其看作一个蓝图或快照。
- **容器：** 镜像的一个可运行实例。您可以启动、停止、移动和删除容器。每个容器都与其他的容器和宿主机隔离运行，但会使用宿主机的操作系统内核，这使得它们与虚拟机相比更加轻量。

对于 LangChain 应用，使用 Docker 意味着您可以将 Python 环境、LangChain 框架、特定库版本（如 `openai`、`tiktoken`、向量 (vector)存储客户端）以及您的自定义链或代理代码打包在一起。这个包随后可以在开发者的笔记本电脑、测试服务器或生产云实例上稳定运行。

### 为 LangChain 应用创建 Dockerfile

`Dockerfile` 是一个文本文件，包含构建 Docker 镜像的逐步指令。以下是 Python 版 LangChain 应用中常用指令的分析：

1. **指定基础镜像 (`FROM`)**
   这是您镜像的起始点。对于 Python 应用，您通常会使用官方的 Python 镜像。在生产环境中，`slim` 变体通常更受青睐，因为它们体积更小。

   ```dockerfile
   # 从特定版本的 Python slim 镜像开始
   FROM python:3.11-slim
   ```
2. **设置工作目录 (`WORKDIR`)**
   此指令设置后续命令（`RUN`、`CMD`、`ENTRYPOINT`、`COPY`、`ADD`）将要执行的目录。它还在您启动容器时定义默认目录。

   ```dockerfile
   # 设置容器内的工作目录
   WORKDIR /app
   ```
3. **复制依赖文件并安装依赖 (`COPY`, `RUN`)**
   最佳做法是首先复制您的依赖文件，并在复制其余应用代码之前安装依赖。这会用到 Docker 的层缓存。如果您的依赖没有改变，Docker 可以复用现有层，从而加快后续构建的速度。

   ```dockerfile
   # 首先复制 requirements 文件以利用 Docker 缓存
   COPY requirements.txt .

   # 安装依赖项
   # 使用 --no-cache-dir 可以减少镜像大小
   RUN pip install --no-cache-dir -r requirements.txt
   ```

   您的 `requirements.txt` 应固定版本以确保可重现性，例如：

   ```txt
   langchain==0.3.0
   langchain-openai==0.2.0
   openai==1.47.0
   python-dotenv==1.0.1
   fastapi==0.115.0 # 如果构建 API
   uvicorn==0.31.0 # 如果构建 API
   # 添加其他必要的库，如向量存储客户端等
   ```
4. **复制应用程序代码 (`COPY`)**
   安装依赖后，将其余的应用程序源代码复制到工作目录中。

   ```dockerfile
   # 复制其余的应用程序代码
   COPY . .
   ```

   为了防止不必要的文件（如 `.git`、`__pycache__`、虚拟环境、本地配置文件）被复制到镜像中，从而可能使其膨胀或泄漏敏感信息，请在与 `Dockerfile` 相同的目录中使用 `.dockerignore` 文件。
5. **设置环境变量 (`ENV`)**
   您可以在 `Dockerfile` 中直接设置环境变量。`PYTHONUNBUFFERED=1` 对于 Python 应用很常见，以确保日志直接发送到终端（并因此被 Docker 日志捕获）而无需缓冲。秘密的占位符通常不应在此处硬编码；它们最好在运行时注入。

   ```dockerfile
   # 确保 Python 输出直接发送到终端
   ENV PYTHONUNBUFFERED=1
   # 示例：如果需要，设置默认配置文件路径
   # ENV CONFIG_FILE=/app/config/production.yaml
   ```
6. **公开端口 (`EXPOSE`)**
   如果您的 LangChain 应用作为 Web 服务运行（例如，使用 FastAPI 或 Flask），`EXPOSE` 指令会告知 Docker 容器在运行时监听指定的网络端口。这主要用于说明；在运行容器时，您仍然需要映射端口。

   ```dockerfile
   # 公开应用程序运行的端口（例如，FastAPI 使用 8000）
   EXPOSE 8000
   ```
7. **定义运行时命令 (`CMD` 或 `ENTRYPOINT`)**
   这指定了从镜像启动容器时要执行的命令。

   - `CMD`：提供默认参数 (parameter)，这些参数在启动容器时可以被覆盖。
   - `ENTRYPOINT`：将容器配置为可执行文件运行。传递给 `docker run` 的参数会附加到 `ENTRYPOINT`。

   对于脚本：

   ```dockerfile
   # 运行应用程序脚本的命令
   CMD: ["python", "main.py"]
   ```

   对于 Uvicorn 这样的 Web 服务器（与 FastAPI 一起使用）：

   ```dockerfile
   # 使用 uvicorn 运行 FastAPI 应用程序的命令
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

   使用 `0.0.0.0` 可以使服务器从容器网络外部访问（当端口被映射时）。

### 构建 Docker 镜像

准备好 `Dockerfile` 和 `.dockerignore` 文件后，在您的终端中进入包含它们 的目录，并运行构建命令：

```bash
docker build -t langchain-prod-app:0.1 .
```

- `docker build`：用于构建镜像的命令。
- `-t langchain-prod-app:0.1`：为镜像打上名称（`langchain-prod-app`）和版本标签（`0.1`）。打标签对版本管理很有用。
- `.`：指定构建上下文 (context)（当前目录），其中包含 `Dockerfile` 和要复制的应用程序代码。

Docker 将逐步执行 `Dockerfile` 中的指令，为每条指令创建层。

### 运行 Docker 容器

要从您刚刚构建的镜像中在容器内部运行 LangChain 应用：

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY="your_actual_api_key" --name my_langchain_instance langchain-prod-app:0.1
```

- `docker run`：创建并启动容器的命令。
- `-p 8000:8000`：将您宿主机器上的 8000 端口映射到容器内部的 8000 端口（在我们的示例中由 Uvicorn 公开）。格式为 `host_port:container_port`。
- `-e OPENAI_API_KEY="your_actual_api_key"`：在容器内部设置环境变量。这是在运行时传递 API 密钥等秘密的常用方式，而不是将它们嵌入 (embedding)到镜像中。我们稍后会介绍更安全的秘密管理方法。
- `--name my_langchain_instance`：为正在运行的容器分配一个易读的名称。
- `langchain-prod-app:0.1`：要运行的镜像的名称和标签。

您现在应该能够与您的 LangChain 应用进行交互（例如，如果它是一个 API，可以通过向 `http://localhost:8000` 发送请求）。

### Docker 化应用的良好做法

- **减小镜像大小：** 较小的镜像拉取和部署速度更快，并且具有更小的攻击面。使用 `.dockerignore`，选择 `slim` 基础镜像，仅安装必要的依赖，使用 `pip` 的 `--no-cache-dir` 参数 (parameter)，并在同一个 `RUN` 指令中清理临时文件。对于在最终运行时镜像中不需要构建工具的复杂情况，可以考虑多阶段构建。
- **优化层缓存：** 构建您的 `Dockerfile`，将不经常更改的指令（如安装依赖）放在经常更改的指令（如复制应用程序代码）之前。
- **固定依赖版本：** 使用带有固定版本（`==`）的 `requirements.txt` 或 `pyproject.toml` 来确保确定性构建。
- **以非 root 用户运行：** 默认情况下，容器以 root 用户身份运行。在您的 `Dockerfile` 中创建一个非 root 用户和组，并在运行应用程序之前使用 `USER` 指令切换到该用户。这会提高安全性。

  ```dockerfile
  # 创建一个非 root 用户和组
  RUN groupadd -r appuser && useradd -r -g appuser appuser

  # 切换到非 root 用户
  USER appuser

  # CMD 或 ENTRYPOINT 在后面
  CMD ["python", "main.py"]
  ```
- **安全处理秘密信息：** 不要将 API 密钥、密码或其他秘密信息硬编码在您的 `Dockerfile` 中，也不要直接将其复制到镜像中。使用在运行时注入的环境变量（如上所示）、Docker secrets 或由您的部署平台提供的配置管理工具（如 Kubernetes Secrets 或云服务商的秘密管理器）。
- **标准化日志记录：** 配置您的应用将日志输出到标准输出 (`stdout`) 和标准错误 (`stderr`)。Docker 会自动收集这些流，方便您使用 `docker logs <container_name_or_id>` 查看日志。

### 示例文件

#### `Dockerfile` (基于 FastAPI 的 LangChain 应用示例)

```dockerfile
# 1. 基础镜像
FROM python:3.11-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 环境变量
ENV PYTHONUNBUFFERED=1

# 4. 首先复制依赖文件以利用缓存
COPY requirements.txt .

# 5. 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 6. 为安全创建一个非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 7. 复制应用程序代码
# 确保之后非 root 用户的权限正确
COPY --chown=appuser:appuser . .

# 8. 公开端口
EXPOSE 8000

# 9. 切换到非 root 用户
USER appuser

# 10. 运行应用程序
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### `.dockerignore` (示例)

```
# Git 文件
.git
.gitignore

# Python 缓存和虚拟环境
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
env/

# 本地配置/秘密文件
.env
*.local

# IDE 文件
.idea/
.vscode/

# 测试文件
.pytest_cache/
htmlcov/
.coverage
```

通过使用 Docker 对 LangChain 应用进行容器化，您创建了一个自包含、可移植且可重现的单元。这个标准化的包简化了应用在不同环境中迁移的过程，为接下来要讨论的部署策略（无论是针对服务器、Kubernetes 集群还是无服务器平台）打下了基础。

## 参考资料

- [What is a container?](https://docs.docker.com/get-started/overview/) — Docker Documentation (2024)
  Publisher: Docker
  解释Docker镜像和容器等概念，以及容器化的基本原理。
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker Documentation (2024)
  Dockerfile指令的详细指南，包括其语法和用法。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  Publisher: Docker
  探讨Dockerfile优化、安全考量以及用于高效镜像构建的层缓存。
- [python - Official Image | Docker Hub](https://hub.docker.com/_/python) — Docker Official Images (2024)
  提供官方Python基础镜像的信息，包括slim版本及其使用方法。

---

[上一节](01-%E4%B8%BA%E9%83%A8%E7%BD%B2%E6%9E%84%E5%BB%BA%20LangChain%20%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.md) · [下一节](03-%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88%EF%BC%9A%E6%9C%8D%E5%8A%A1%E5%99%A8%E3%80%81Kubernetes%E3%80%81%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8.md)
