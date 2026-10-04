# 使用 Docker 容器化应用

来源：[原文](https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals/containerizing-with-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

部署机器学习 (machine learning)模型通常涉及构建 REST API 来提供预测服务。虽然这是使模型变得有用的一个主要步骤，但在将此类 API 从开发机迁移到测试或生产服务器时，常常会出现一个挑战。确保应用程序在任何地方都能以完全相同的方式运行可能很困难。操作系统、已安装的库版本或系统配置的差异经常导致意外错误——这就是广为人知的“在我的机器上能运行”问题。

正因如此，容器化，尤其是使用 Docker，变得非常有价值。Docker 让您能够将应用程序及其所有依赖项（库、运行时、系统工具）打包成一个称为容器的标准化单元。此容器包含运行应用程序所需的一切，确保在不同环境中保持一致性。

### Docker 是什么？

可以将 Docker 容器视为一个轻量、独立、可执行的软件包。它将您的代码（您的模型预测 API 脚本）、Python 运行时、必需的库（如 Flask/FastAPI、scikit-learn、pandas）以及任何所需的系统设置捆绑在一起。与打包整个操作系统的传统虚拟机（VM）不同，容器共享宿主系统的操作系统内核。这使得它们比虚拟机更小巧、启动更快、资源效率更高。

常用的比喻是海运集装箱。在标准化海运集装箱出现之前，将各种形状和尺寸的货物装载到船上既复杂又低效。海运集装箱使流程标准化，极大简化了运输。Docker 对软件也做了类似的事情：它提供了一种标准方法来打包和运行应用程序，从而简化了部署。

### Docker 核心原理

要使用 Docker，您需要了解一些基本原理：

1. **Dockerfile：** 这是一个文本文件，包含如何构建 Docker 镜像的一系列指令。它充当您容器化应用程序的蓝图。您在此文件中定义基础环境、复制代码、安装依赖项，并指定应用程序如何运行。
2. **镜像：** 镜像是从 Dockerfile 创建的只读模板。它包含应用程序代码、库、依赖项、工具以及应用程序运行所需的其他文件。镜像通常分层构建，这使得它们存储和分发时效率高。
3. **容器：** 容器是镜像的一个可运行实例。当您运行 Docker 镜像时，会创建一个容器。您可以基于镜像创建、启动、停止、移动或删除容器。容器是实际的、活动的、正在运行的应用程序环境。

### 为您的模型 API 创建 Dockerfile

我们来为上一节中构建的 REST API 创建一个 `Dockerfile`。假设您的 API 代码在一个名为 `app.py` 的文件中，并且您的依赖项列在 `requirements.txt` 中，一个典型的 `Dockerfile` 可能如下所示：

```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置容器中的工作目录
WORKDIR /app

# 将 requirements 文件复制到容器的 /app 目录
COPY requirements.txt .

# 安装 requirements.txt 中指定的任何必需包
# --no-cache-dir 通过不存储 pip 缓存来减小镜像大小
# --trusted-host pypi.python.org 避免在某些网络中潜在的 SSL 问题
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# 将其余应用程序代码复制到容器的 /app 目录
# 这包括 app.py、您保存的模型文件（例如 model.pkl）等
COPY . .

# 使端口 8000 可从此容器外部访问（如果使用 FastAPI，如果使用 Flask 则调整为例如 5000）
EXPOSE 8000

# 定义环境变量（可选，可能有用）
# ENV MODEL_NAME=my_model.pkl

# 容器启动时运行 app.py
# 对于 FastAPI 使用 uvicorn，对于 Flask 使用 python app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Flask 示例：CMD ["python", "app.py"]
```

我们来细读这些指令：

- `FROM python:3.9-slim`：指定要使用的基础镜像。我们从官方 Python 3.9 镜像的一个精简版本开始。使用特定版本标签（如 `3.9-slim` 而不是 `python:latest`）可确保复现性。
- `WORKDIR /app`：将容器内的工作目录设置为 `/app`。后续命令（如 `COPY` 和 `RUN`）将相对于此目录执行。
- `COPY requirements.txt .`：将 `requirements.txt` 文件从您的本地目录（构建上下文 (context)）复制到容器的工作目录（`/app`）。
- `RUN pip install ...`：执行命令以安装 `requirements.txt` 中列出的 Python 依赖项。`--no-cache-dir` 标志可防止 pip 存储下载缓存，有助于减小镜像大小。在特定网络环境中，有时可能需要 `--trusted-host`。
- `COPY . .`：将构建上下文（包含 `Dockerfile`、`app.py`、`model.pkl` 等的您的项目目录）中的所有文件和目录复制到容器的工作目录（`/app`）。在您的项目目录中有一个 `.dockerignore` 文件是一个好习惯，可以排除不必要的文件（如虚拟环境、`.git` 目录、缓存文件）不被复制，从而进一步减小镜像大小。
- `EXPOSE 8000`：通知 Docker 容器在运行时监听网络端口 8000。这实际上不会发布端口；它充当文档，并被某些编排工具使用。
- `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`：指定从此镜像启动容器时要执行的默认命令。在这里，它使用 `uvicorn` 启动 FastAPI 应用程序，并将其绑定到容器内部端口 8000 上的所有网络接口（`0.0.0.0`）。根据您的 Web 框架调整此命令（例如，对于简单的 Flask 应用程序，使用 `CMD ["python", "app.py"]`）。

### 构建 Docker 镜像

拥有 `Dockerfile` 后，在您的终端中导航到您的项目目录（包含 `Dockerfile`、`app.py`、`requirements.txt` 等的目录），并运行构建命令：

```bash
docker build -t your-model-api:v1 .
```

- `docker build`：从 Dockerfile 构建镜像的命令。
- `-t your-model-api:v1`：为镜像添加名称（`your-model-api`）和版本标签（`v1`）。打标签便于管理和引用镜像。选择一个有意义的名称和标签。
- `.`：指定构建上下文 (context)——当前目录。Docker 将在此处查找 `Dockerfile`，并将目录内容发送给 Docker 守护程序进行构建。

Docker 将逐步执行 `Dockerfile` 中的指令，可能下载基础镜像并安装依赖项。完成后，您将在本地存储一个名为 `your-model-api:v1` 的 Docker 镜像。您可以使用 `docker images` 查看您的镜像。

### 运行 Docker 容器

现在您有了镜像，可以将其作为容器运行：

```bash
docker run -p 8080:8000 your-model-api:v1
```

- `docker run`：从镜像创建并启动容器的命令。
- `-p 8080:8000`：将您宿主机上的端口 8080 映射到容器内部的端口 8000（`uvicorn`/`Flask` 暴露的端口）。这使您可以通过宿主机上的 `http://localhost:8080` 访问在容器内运行的 API。如果端口 8080 已被占用，您可以选择不同的宿主端口（例如，`-p 5001:8000`）。
- `your-model-api:v1`：要运行的镜像的名称和标签。

您的 API 现在应该在 Docker 容器中运行了！您可以使用 `curl`、Postman 或您的网络浏览器等工具，向 `http://localhost:8080`（或您映射的任何宿主端口）发送请求来测试它。

要在后台（分离模式）运行容器，请添加 `-d` 标志：

```bash
docker run -d -p 8080:8000 your-model-api:v1
```

这将打印一个容器 ID 并将控制权返回给您的终端。您可以使用 `docker ps` 查看正在运行的容器。要停止一个分离的容器，请使用 `docker stop <container_id_or_name>`。

### 容器化模型 API 的优点

使用 Docker 容器化您的模型 API 带来多项益处：

- **一致性：** 确保无论容器在何处运行，都使用完全相同的环境（Python 版本、库版本），从而消除与环境相关的部署问题。
- **复现性：** 将应用程序及其依赖项一起打包，使得其他人（或自动化系统）可以轻松可靠地设置和运行 API。
- **隔离性：** API 在隔离环境中运行，防止与宿主系统上的其他应用程序或库发生冲突。
- **可移植性：** 容器镜像可以在任何安装了 Docker 的机器上运行，无论是您的笔记本电脑、同事的机器、本地服务器还是云平台（如 AWS、GCP、Azure）。
- **可扩展性：** 容器编排工具（如 Kubernetes 或 Docker Swarm）使运行和管理容器化 API 的多个实例以处理增加的负载变得简单。

### 机器学习 (machine learning)模型的相关考量

- **模型文件：** 在示例 `Dockerfile` 中，我们使用 `COPY . .`，它将所有内容（包括可能很大的模型文件，如 `.pkl`、`.h5` 等）直接复制到镜像中。这很简单，并确保模型始终与代码一起打包。然而，这会增加镜像大小。对于非常大的模型或在不更改代码的情况下频繁更新模型，存在替代方案，例如在运行时挂载卷（`docker run -v /path/on/host:/path/in/container ...`），但这会增加复杂性。对于此阶段的大多数典型情况，将模型复制到镜像中是足够的。
- **镜像大小：** 保持镜像小巧是好的做法。使用 `.dockerignore` 文件排除不必要的文件，使用精简的基础镜像（如 `python:3.9-slim`），并在构建期间清理不必要的文件（例如，使用 `--no-cache-dir`）。更高级的方法，如多阶段构建，可以进一步优化大小。
- **安全性：** 使用官方基础镜像，如果可能，避免在容器内以 root 身份运行，并注意任何可能复制到镜像中的敏感信息。

使用 Docker 容器化您的应用程序是现代软件和机器学习部署的一项基本技能。它提供了一种可移植且一致的方法来打包和运行您的模型服务 API，弥合了开发和生产环境之间的差距。现在您的 API 已容器化，下一步是考虑如何在其部署后监控其性能和状况。

## 参考资料

- [Docker overview](https://docs.docker.com/get-started/overview/) — Docker Documentation (2023)
  解释 Docker 的基本概念、架构和术语，作为容器化的权威入门指南。
- [Python Official Image](https://hub.docker.com/_/python) — Docker Official Images (2023)
  提供官方 Python Docker 镜像的详细信息和使用说明，对容器化基于 Python 的机器学习应用程序至关重要。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.abebooks.com/book-search/isbn/9781492045113/) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  讨论部署机器学习模型的更广泛背景，重点介绍容器化作为 MLOps 框架中的核心部署策略。

---

[上一节](03-%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E7%9A%84REST%20API.md) · [下一节](05-%E6%A8%A1%E5%9E%8B%E7%9B%91%E6%8E%A7%E5%9F%BA%E6%9C%AC%E8%A6%81%E7%82%B9.md)
