# 动手实践：通过 Docker 部署 LangChain 应用

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production/practice-deploying-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本次动手实践将指导您使用 Docker 打包一个简单的 LangChain 应用程序。容器化提供了一个一致、隔离的环境，使部署在不同机器和平台之间可预测。这种一致性对于生产系统来说很关键。目标是为基础的 LangChain 应用程序创建一个 Docker 镜像，并在本地作为容器运行它。

### 先决条件

开始之前，请确保您已安装以下各项：

1. **Docker:** Docker Desktop（适用于 Windows/macOS）或 Docker Engine（适用于 Linux）。您可以通过在终端中运行 `docker --version` 来验证您的安装。
2. **Python:** 一个可用的 Python 3.9+ 安装。
3. **文本编辑器/IDE:** 您喜欢的代码文件编辑工具。

### 步骤 1: 创建一个简单的 LangChain 应用程序

首先，让我们创建一个最简单的 LangChain 应用程序。我们将构建一个简单的链，它接收一个主题并使用 LLM 生成简要解释。

为您的项目创建一个目录，例如 `langchain_docker_app`。在此目录中，创建两个文件：`app.py` 和 `requirements.txt`。

**`requirements.txt`:**

```txt
langchain>=0.3.0
langchain-openai>=0.2.0
python-dotenv>=1.0.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
```

*注意: 请根据您打算使用的 LLM 调整 LLM 提供方库（例如 `langchain-google-genai`、`langchain-anthropic`）。*

**`app.py`:**

我们将使用 `FastAPI` 提供应用程序服务，并使用 `Pydantic` 进行数据验证，这是生产 API 的推荐做法。

```python
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# 加载环境变量（特别是 API 密钥）
load_dotenv()

# 确保您的 OPENAI_API_KEY 已在 .env 文件或环境变量中设置
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title="简单的 LangChain API",
    description="一个使用 Docker 呈现 LangChain 的基础 API。",
)

# 2. 定义请求模型
class TopicRequest(BaseModel):
    topic: str

# 3. 设置 LangChain 组件
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手，可以简单地解释技术想法。"),
        ("human", "用一句话解释 '{topic}' 这个想法。"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
output_parser = StrOutputParser()

# 4. 使用 LCEL 创建链
chain = (
    {"topic": RunnablePassthrough()} # 直接将输入作为 'topic' 传递
    | prompt_template
    | llm
    | output_parser
)

# 5. 定义 API 端点
@app.post("/explain")
async def explain_topic(request: TopicRequest):
    """
    接受一个带有 'topic' 键的 JSON 对象，并返回一个解释。
    示例：{"topic": "Docker"}
    """
    try:
        # 使用请求模型中的主题调用链
        result = chain.invoke(request.topic)
        return {"explanation": result}
    except Exception as e:
        # 基础错误处理
        raise HTTPException(status_code=500, detail=str(e))

# 添加一个根端点用于基本检查
@app.get("/")
async def read_root():
    return {"message": "LangChain API 正在运行。"}

# 注意：要在本地运行此程序（不先使用 Docker）
# 您通常会使用：uvicorn app:app --reload --port 8000
# 记得创建一个包含您的 OPENAI_API_KEY 的 .env 文件
```

在同一目录中创建一个 `.env` 文件以安全地存储您的 API 密钥：

**`.env`:**

```
OPENAI_API_KEY=your_openai_api_key_here
```

将 `your_openai_api_key_here` 替换为您的实际 API 密钥。

### 步骤 2: 创建 Dockerfile 和 .dockerignore

为确保我们构建一个干净且安全的镜像，我们必须排除本地配置文件（如 `.env`）和虚拟环境。将敏感信息复制到镜像中是一个严重的安全性风险。在项目目录中创建一个名为 `.dockerignore` 的文件：

**`.dockerignore`:**

```text
.env
__pycache__
.git
venv/
.DS_Store
```

现在，创建 `Dockerfile`（无扩展名）以定义构建指令。

**`Dockerfile`:**

```dockerfile
# 1. 使用官方 Python 运行时作为父镜像
# 使用 slim 版本可减小镜像大小
FROM python:3.11-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 将 requirements 文件复制到容器的 /app 目录
# 先复制 requirements 文件以使用 Docker 缓存
COPY requirements.txt .

# 4. 安装 requirements.txt 中指定的任何所需软件包
# --no-cache-dir 减少层大小
# --upgrade pip 确保我们拥有最新版本的 pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 将应用程序的其余代码复制到容器的 /app 目录
# .dockerignore 中列出的文件（如 .env）将被排除
COPY . .

# 6. 使容器外部可访问端口 8000
# 这是 Uvicorn 将运行的端口
EXPOSE 8000

# 7. 容器启动时运行 app.py
# 使用 uvicorn 运行 FastAPI 应用程序
# --host 0.0.0.0 使其可从容器外部访问
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

我们来分析一下这些指令：

- `FROM python:3.11-slim`: 指定基础镜像。使用 slim 版本有助于减小最终镜像的大小。
- `WORKDIR /app`: 设置容器内部的工作目录。后续命令（`COPY`、`RUN`、`CMD`）将相对于此目录执行。
- `COPY requirements.txt .`: 将 requirements 文件复制到 `/app` 目录。我们先复制此文件以使用 Docker 的层缓存。如果 `requirements.txt` 没有改变，Docker 会重复使用安装依赖项的层，从而加快后续构建的速度。
- `RUN pip install ...`: 安装 Python 依赖项。`--no-cache-dir` 阻止 pip 存储下载内容，减小镜像大小。
- `COPY . .`: 将项目其余文件复制到 `/app` 目录。因为我们创建了 `.dockerignore` 文件，所以敏感文件（如 `.env`）将**不会**被复制。
- `EXPOSE 8000`: 通知 Docker 容器在运行时监听端口 8000。
- `CMD ["uvicorn", ...]`: 指定容器启动时运行的命令。这里，我们启动 Uvicorn 服务器来提供 FastAPI 应用程序，并将其绑定到 `0.0.0.0`，以便可以从容器的网络命名空间外部访问它。

### 步骤 3: 构建 Docker 镜像

在终端中进入您的 `langchain_docker_app` 目录。运行以下命令来构建 Docker 镜像：

```bash
docker build -t langchain-simple-api:latest .
```

- `docker build`: 用于从 Dockerfile 构建镜像的命令。
- `-t langchain-simple-api:latest`: 为镜像打上名称 (`langchain-simple-api`) 和标签 (`latest`)。这使得之后引用镜像更加便捷。
- `.`: 指定构建上下文 (context)（当前目录），指示 Docker 应在哪里查找 `Dockerfile` 和应用程序文件。

Docker 将逐步执行 `Dockerfile` 中的指令。您将看到显示每个步骤进度的输出。

### 步骤 4: 运行 Docker 容器

镜像构建成功后，您可以将其作为容器运行：

```bash
docker run -p 8000:8000 --env-file .env --name my-langchain-container langchain-simple-api:latest
```

我们来分析此命令：

- `docker run`: 用于从镜像创建并启动容器的命令。
- `-p 8000:8000`: 将主机上的端口 8000 映射到容器内部的端口 8000。这允许您通过 `http://localhost:8000` 在您的机器上访问容器内运行的 FastAPI 应用程序。格式为 `host_port:container_port`。
- `--env-file .env`: 将指定 `.env` 文件中的环境变量加载到容器中。这是一种安全的方法，可以在不将 API 密钥和其他配置硬编码到 `Dockerfile` 或镜像中的情况下进行传递。
- `--name my-langchain-container`: 为正在运行的容器指定一个名称，以便于管理（例如，停止或查看日志）。
- `langchain-simple-api:latest`: 指定用于创建容器的镜像。

如果您想让容器在后台运行（分离模式），请添加 `-d` 标志：

```bash
docker run -d -p 8000:8000 --env-file .env --name my-langchain-container langchain-simple-api:latest
```

### 步骤 5: 验证应用程序

容器运行后，您可以验证应用程序：

1. **检查容器日志（特别是在分离模式下运行）**：

   ```bash
   docker logs my-langchain-container
   ```

   您应该看到 Uvicorn 的输出，表明服务器已启动。
2. **访问根端点**：打开您的网页浏览器或使用 `curl`：

   ```bash
   curl http://localhost:8000/
   ```

   您应该收到：`{"message":"LangChain API 正在运行。"}`
3. **测试 `/explain` 端点**：使用 `curl` 或 Postman 等工具发送 POST 请求：

docker run -p 8000:8000 --env-file .env --name my-langchain-container langchain-simple-api:latest

````
您应该收到一个 JSON 响应，其中包含由 LangChain 链生成的 Kubernetes 解释，例如：
    ```json
    {"explanation": "Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications."}
    ```

### 步骤 6: 停止和移除容器

要停止在前台运行的容器，请在运行它的终端中按 `Ctrl+C`。如果在分离模式下运行，请使用：

```bash
docker stop my-langchain-container
````

要删除已停止的容器（可选，释放名称和磁盘空间）：

```bash
docker rm my-langchain-container
```

### 总结和下一步

您已成功地使用 FastAPI 将一个简单的 LangChain 应用程序打包到 Docker 容器中并在本地运行。这说明了容器化的核心工作流程：即在 `Dockerfile` 中定义环境和依赖项，构建可移植的镜像，并可预测地运行它。

这个容器化应用程序构成了部署到前面提到的各种环境的根基，例如虚拟机、Kubernetes 集群或无服务器平台。生产工作流中下一个合乎逻辑的步骤是将这个构建好的镜像推送到容器注册表（如 Docker Hub、AWS ECR、Google Artifact Registry 或 Azure Container Registry），部署环境可以从中拉取并运行它。本次练习提供了准备 LangChain 应用程序部署所需的必要技能。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  提供所有 Docker 组件的官方指南和参考资料，包括 Dockerfile、镜像构建、容器运行以及基本的容器化原则。
- [LangChain Documentation: LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Team (2024)
  Publisher: LangChain
  LCEL 的官方指南，该指南用于在实践示例中构建 LangChain 应用程序，并详细介绍了如何组合链和可运行组件。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  Publisher: FastAPI
  FastAPI 的官方文档，提供了在 Python 中构建 API 的快速高效方法，如示例应用程序的端点所示。

---

[上一节](07-%E8%93%9D%E7%BB%BF%E9%83%A8%E7%BD%B2%E4%B8%8E%E9%87%91%E4%B8%9D%E9%9B%80%E9%83%A8%E7%BD%B2%E7%AD%96%E7%95%A5.md) · [下一节](../08-LangChain%20%E5%BA%94%E7%94%A8%E4%B8%AD%E7%9A%84%E5%AE%89%E5%85%A8%E8%80%83%E9%87%8F/01-%E8%AE%A4%E8%AF%86LLM%E5%BA%94%E7%94%A8%E4%B8%AD%E7%9A%84%E6%94%BB%E5%87%BB%E9%80%94%E5%BE%84.md)
