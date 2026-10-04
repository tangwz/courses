# 在 Docker 镜像中打包机器学习模型

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/including-ml-models-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将机器学习 (machine learning)模型作为 Docker 容器镜像的一部分，是 FastAPI 应用部署的主要考量。为了让 API 独立运行，它需要访问容器文件系统内的已训练模型文件（例如序列化模型文件、权重 (weight)或配置文件）。以下是实现此目的的常见策略。

### 方法一：直接复制本地模型文件

尤其是在开发阶段或针对简单项目，最直接的方法就是将模型文件直接包含在您的项目结构中，并在构建过程中将其复制到镜像内。

假设您的项目布局如下所示：

```
.
├── app
│   ├── main.py
│   └── ml_model.py
├── models
│   └── model_v1.joblib  # 您的序列化模型
├── requirements.txt
└── Dockerfile
```

在这种情况下，您的 `Dockerfile` 可以使用 `COPY` 指令将模型文件放到镜像内指定的位置：

```dockerfile
# 使用 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 首先复制依赖文件，利用 Docker 缓存
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app /app/app

# --- 添加此部分以复制模型 ---
# 在容器内创建模型目录
RUN mkdir /app/models
# 将模型文件从主机复制到镜像中
COPY ./models/model_v1.joblib /app/models/model_v1.joblib
# --- 模型复制部分结束 ---

# 运行应用的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

在您的 FastAPI 应用（`app/ml_model.py` 或类似文件）中，您将使用 `Dockerfile` 中指定的路径（`/app/models/model_v1.joblib`）加载模型。

**优点：**

- **简单易懂：** 易于理解和实现。
- **自包含镜像：** 生成的 Docker 镜像包含运行应用所需的一切，包括构建时使用的特定模型版本。这有助于实现更好的可复现性。

**缺点：**

- **镜像大小：** 机器学习 (machine learning)模型可能很大（数百 MB 甚至数 GB）。直接包含它们会大幅增加 Docker 镜像的大小，影响存储、传输时间和可能的冷启动时间。
- **高耦合度：** 每次更新模型文件时，即使应用代码没有改变，也需要重新构建 Docker 镜像。
- **仓库大小：** 通常不建议将大型模型文件直接存储在 Git 仓库中（如果必须，可以考虑 Git LFS，但这会增加操作复杂性）。

### 方法二：在构建时下载模型

另一种方法是在 Docker 镜像构建过程中从外部来源下载模型。这个来源可以是云存储（如 AWS S3、Google Cloud Storage、Azure Blob Storage）、专门的模型注册表，甚至是一个简单的 HTTP URL。

您将在 `Dockerfile` 中使用 `RUN` 指令，并配合 `curl`、`wget` 或云服务特定的命令行工具。

```dockerfile
# 使用 Python 基础镜像
FROM python:3.10-slim

# 如果基础镜像中没有，安装必要的工具（例如 curl）
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* # Debian 系镜像示例

# 设置工作目录
WORKDIR /app

# 首先复制依赖文件
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app /app/app

# --- 添加此部分以下载模型 ---
# 定义模型 URL（也可以使用 ARG 增加灵活性）
ARG MODEL_URL="https://your-storage-provider.com/models/model_v1.joblib"
# 创建模型目录
RUN mkdir /app/models
# 使用 curl 下载模型
RUN curl -o /app/models/model_v1.joblib ${MODEL_URL}
# --- 模型下载部分结束 ---

# 运行应用的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

您可以在构建镜像时将 `MODEL_URL` 作为构建参数 (parameter)传递：

```bash
docker build --build-arg MODEL_URL="<actual_model_url>" -t my-ml-api .
```

**优点：**

- **分离存储：** 模型文件与应用代码仓库分开存储。
- **更小的仓库：** 您的 Git 仓库不需要存储大型模型文件。
- **更强的适应性：** 通过更改构建参数或源 URL，可以更容易地切换模型版本，而无需大幅修改 `Dockerfile` 的核心逻辑。

**缺点：**

- **构建依赖：** Docker 构建过程现在依赖于外部模型存储的可用性和可访问性。网络问题或存储中断可能导致构建失败。
- **构建时间：** 下载大型模型会大幅增加镜像构建时间。
- **安全问题：** 如果模型存储需要认证，您需要在构建过程中提供凭证的安全方式。Docker 构建秘密或多阶段构建有助于管理此问题，防止凭证泄露到最终的镜像层中。
- **构建工具：** 您可能需要在镜像中安装额外的工具（如 `curl`、`awscli`、`gcloud-sdk` 等），仅仅是为了下载步骤，这可能会增加镜像大小，除非您使用多阶段构建在后续步骤中丢弃这些工具。

### 方法三：通过卷挂载模型（部署时请谨慎使用）

Docker 卷允许您在运行时将主机上的目录或受管理的 Docker 卷挂载到容器中。虽然这对本地开发很有用（例如，无需重新构建镜像即可快速测试不同模型），但在生产部署场景中使用卷来提供主要模型通常是**不**建议的。

它违背了拥有自包含、不可变镜像的原则。容器的功能将依赖于运行时正确挂载的外部文件系统，这会使部署编排和可复现性变得复杂。如果卷未正确挂载，应用将失败。这种方法更适合注入配置或运行时数据，而不是模型本身等核心应用文件。

### 考量因素与实用建议

- **模型版本管理：** 明确管理包含哪个模型版本。
  - **直接复制：** 使用清晰的文件名（例如 `model_v1.2.joblib`）。更新模型需要更改 `COPY` 指令中的文件名并替换文件。
  - **下载：** 使用构建参数 (parameter)（`ARG`）指定模型版本或 URL，以便在 CI/CD 流水线中更方便地参数化构建。
- **文件路径：** 确保路径一致。模型在 `Dockerfile` 中保存的路径（例如 `/app/models/`）必须与您的 FastAPI 应用代码加载模型时使用的路径匹配。在应用内部使用环境变量来指定模型路径可以增加其适应性。
- **镜像大小优化：**
  - 如果下载，考虑使用多阶段构建。一个阶段使用所需的工具下载模型，而最终的、更小的阶段只复制应用代码和已下载的模型，丢弃下载工具。
  - 确保您的基础 Python 镜像足够小（例如 `python:3.X-slim`）。
  - 清理包管理器缓存（例如 Debian/Ubuntu 的 `apt-get clean`、`rm -rf /var/lib/apt/lists/*`；pip 的 `--no-cache-dir`）。
- **下载安全：** 如果从受保护的存储下载，请使用安全方法，例如通过构建参数注入短期凭证（谨慎使用）或 Docker 的构建秘密管理功能（`--secret` 标志），以避免将敏感信息嵌入 (embedding)到镜像层中。

选择正确的方法取决于您的项目复杂程度、团队工作流程、模型大小和部署环境。对于许多应用，直接复制模型可提供简单性，而在构建时下载则为更复杂或频繁更新并单独存储的模型提供了更好的独立性。

## 参考资料

- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  Publisher: Docker
  介绍了 Dockerfile 的通用原则、镜像优化、缓存策略和多阶段构建，这对于高效地容器化机器学习模型至关重要。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready ML Applications](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供了部署和管理机器学习模型的架构考量，包括在生产环境中进行版本控制和工件处理的策略。
- [Docker BuildKit features: Build secrets](https://docs.docker.com/build/buildkit/define-secrets/) — Docker Documentation (2024)
  Publisher: Docker
  解释了如何在 Docker 镜像构建过程中安全地传递敏感信息（如凭据），而无需将其嵌入到镜像层中，解决了从受保护源下载模型时的安全问题。

---

[上一节](02-%E7%BC%96%E5%86%99%20FastAPI%20%E5%BA%94%E7%94%A8%E7%9A%84%20Dockerfile.md) · [下一节](04-%E6%9E%84%E5%BB%BA%E5%92%8C%E8%BF%90%E8%A1%8C%20Docker%20%E5%AE%B9%E5%99%A8.md)
