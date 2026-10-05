# 动手实践：一个简单LLM应用的容器化

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations/practice-containerizing-llm-app)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LLM应用需要打包以进行部署，这是开发过程中的重要一步。容器化，主要是使用Docker，是一种标准做法，可帮助确保你的应用在从开发机到测试服再到生产服等不同环境中都能稳定运行。

本动手练习将引导你完成一个简单LLM应用的容器化过程，类似于你可能在前面章节中构建的问答机器人或代理。

**前提条件**

在开始之前，请确保你已具备：

1. **Docker已安装：** 你的系统需要安装并运行Docker Desktop（适用于Mac/Windows）或Docker Engine（适用于Linux）。你可以在Docker官方网站上找到安装说明。
2. **一个简单LLM应用：** 准备一个基本的Python应用。这可以是一个简单的Flask或FastAPI Web服务器，其端点接受用户查询，与LLM API（如OpenAI的）交互，并返回结果。假设你的应用结构如下所示：

   ```
   llm_app/
   ├── app.py         # 你的主应用脚本（例如，Flask/FastAPI）
   ├── requirements.txt # 列出Python依赖项（例如，openai, flask）
   └── .env           # 可选：用于本地存储环境变量的文件（不应提交）
   ```

   你的`requirements.txt`可能包含：

   ```
   flask
   openai
   python-dotenv
   ```

   你的`app.py`通常会从环境变量中加载API密钥，并定义一个路由（例如，`/ask`）来处理请求。

**步骤1：创建Dockerfile**

`Dockerfile`是一个文本文件，包含Docker用来构建应用镜像的指令。在你的`llm_app`项目的根目录下创建一个名为`Dockerfile`（无扩展名）的文件。

将以下内容添加到你的`Dockerfile`中：

```dockerfile
# 使用官方Python运行时作为基础镜像
FROM python:3.10-slim

# 设置容器中的工作目录
WORKDIR /app

# 将requirements文件复制到容器的/app目录中
COPY requirements.txt .

# 安装requirements.txt中指定的任何所需包
# 使用--no-cache-dir来减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 将其余应用代码复制到容器的/app目录中
COPY . .

# 使端口5000在容器外部可用（如果使用Flask默认端口）
# 如果你的应用使用不同端口，请调整
EXPOSE 5000

# 定义API密钥的环境变量（占位符）
# 重要提示：这对于生产环境不安全。请参阅下面的说明。
ENV OPENAI_API_KEY=your_default_or_placeholder_key 

# 容器启动时运行应用的命令
# 根据你运行应用的方式（例如，FastAPI使用uvicorn）调整此命令
CMD ["python", "app.py"]
```

**理解Dockerfile指令：**

- `FROM python:3.10-slim`: 指定基础镜像。我们使用的是官方Python 3.10镜像的精简版本。
- `WORKDIR /app`: 设置容器内后续指令的默认目录。
- `COPY requirements.txt .`: 首先只复制requirements文件。Docker会缓存层，因此如果`requirements.txt`没有改变，`pip install`步骤在后续构建时将无需再次运行，从而加快速度。
- `RUN pip install ...`: 执行安装依赖项的命令。
- `COPY . .`: 将你的其余应用代码（如`app.py`）复制到容器中的`/app`目录。
- `EXPOSE 5000`: 通知Docker容器在运行时监听端口5000。这实际上并未发布端口；它更像是一种文档。
- `ENV OPENAI_API_KEY=...`: 在容器镜像*内部*设置一个环境变量。**警告：** 像这样直接硬编码密钥，甚至通过`ENV`设置它们，在生产环境中通常是不安全且不良的做法。它会使密钥在镜像层中可见。我们在此处这样做是为了简便，但在实际应用中，你应在运行时注入密钥（如步骤3所示）或使用密钥管理工具。请返回查阅“管理API密钥和秘密信息”部分，以了解更好的方法。
- `CMD ["python", "app.py"]`: 指定从此镜像启动容器时要执行的默认命令。

**步骤2：构建Docker镜像**

打开你的终端或命令提示符，导航到包含你的`Dockerfile`和应用代码（`llm_app/`）的目录，然后运行构建命令：

```bash
docker build -t my-llm-app .
```

- `docker build`: 构建镜像的命令。
- `-t my-llm-app`: 为镜像打上名称（`my-llm-app`）标签，并可选择一个标记 (token)（默认为`latest`）。这使得以后引用该镜像更加方便。
- `.`: 指定构建上下文 (context)（当前目录），其中包含`Dockerfile`和你的应用代码。

Docker将一步一步地执行你`Dockerfile`中的指令，下载基础镜像并为每条指令创建层。

**步骤3：运行Docker容器**

镜像成功构建后，你可以基于它运行一个容器：

```bash
docker run -p 5001:5000 --env OPENAI_API_KEY="YOUR_ACTUAL_API_KEY" --name llm_container my-llm-app
```

- `docker run`: 创建并启动容器的命令。
- `-p 5001:5000`: 将你主机上的端口5001映射到容器内部的端口5000（你的应用正在监听的端口，由`EXPOSE`和你可能设置的Flask/FastAPI指定）。你现在可以通过`http://localhost:5001`访问你的应用。
- `--env OPENAI_API_KEY="YOUR_ACTUAL_API_KEY"`: 这比通过`ENV`将其内置到镜像中更安全。它仅在运行时为这个特定的容器实例设置环境变量。将`"YOUR_ACTUAL_API_KEY"`替换为你的真实密钥。
- `--name llm_container`: 为正在运行的容器分配一个易于识别的名称。
- `my-llm-app`: 要运行的镜像名称。

如果你想让容器在后台运行（分离模式），请添加`-d`标志：

```bash
docker run -d -p 5001:5000 --env OPENAI_API_KEY="YOUR_ACTUAL_API_KEY" --name llm_container my-llm-app
```

你可以使用`docker logs llm_container`查看日志。要停止容器，请使用`docker stop llm_container`。要删除它，请使用`docker rm llm_container`。

**步骤4：测试容器化应用**

容器运行后，打开一个新终端或使用Postman或`curl`等工具，向你的应用发送请求，该应用现在可以通过你映射的主机端口（例如，5001）访问：

```bash
# 使用curl的示例，假设你的应用在/ask处有一个POST端点
curl -X POST http://localhost:5001/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "以简单的术语解释容器化。"}'
```

你应该收到由LLM生成的响应，该响应由在Docker容器内运行的应用提供。

**优点和后续步骤**

你已成功容器化你的简单LLM应用！这个包（Docker镜像）包含你的应用、其依赖项和运行时，确保在任何有Docker的地方都能稳定运行。这简化了与他人共享你的应用、将其部署到云平台以及可靠地管理依赖项。

对于更复杂的场景，可以考虑以下内容：

- **.dockerignore:** 创建一个`.dockerignore`文件（语法类似于`.gitignore`），以便将文件和目录（如`.git`、`__pycache__`、`.env`）排除在镜像复制之外，使其更小更安全。
- **多阶段构建：** 在你的`Dockerfile`中使用多个`FROM`语句，将构建时依赖项与运行时依赖项分离，从而生成更小的最终镜像。
- **生产密钥管理：** 调查诸如Docker Secrets、Kubernetes Secrets或云提供商服务（例如，AWS Secrets Manager、Google Secret Manager）等工具，用于在生产环境中处理API密钥等敏感数据。

容器化是现代应用开发和部署的基本技能，其中包括使用大型语言模型构建的应用。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  权威指南，详细说明创建Dockerfile的语法和指令，对于将应用程序打包成Docker镜像至关重要。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker (2024)
  Publisher: Docker
  提供构建高效、安全、可维护Docker镜像的实用建议和模式，包含缓存和镜像大小优化的策略。
- [Manage sensitive data with Docker secrets](https://docs.docker.com/engine/swarm/secrets/) — Docker (2024)
  说明如何利用Docker的原生秘密管理功能处理API密钥等敏感信息，相较于将其嵌入环境变量是更安全的方法。

---

[上一节](06-%E7%AE%80%E5%8D%95%E9%83%A8%E7%BD%B2%E9%80%89%E9%A1%B9%EF%BC%88%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E3%80%81%E5%AE%B9%E5%99%A8%EF%BC%89.md)
