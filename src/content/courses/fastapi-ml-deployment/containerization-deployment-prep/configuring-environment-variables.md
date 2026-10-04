---
course: "fastapi-ml-deployment"
chapter: "containerization-deployment-prep"
lesson: "configuring-environment-variables"
sourceId: 5352
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/configuring-environment-variables"
title: "使用环境变量配置应用程序"
description: "在 Docker 容器中，使用环境变量来配置应用程序设置。"
order: 6
plots: []
sourceHash: "abba6a1acf45fba46368350c1522b53f9308344602a95cea4e7f40353628053f"
sourceCorrections: []
---

将 FastAPI 应用程序打包成 Docker 容器，就创建了一个独立的单元。然而，应用程序在不同环境中的运行方式很少完全相同。你可能需要针对开发和生产使用不同的数据库连接字符串，用于外部服务的不同 API 密钥，或者只是一个不同的日志级别来调试。将这些配置值直接硬编码到应用程序代码中会缺乏灵活性，并带来安全风险，特别是对于敏感信息。

在容器化环境中，一种标准且有效的方法是通过**环境变量**来管理配置。这些变量存在于应用程序代码之外，位于应用程序运行的操作系统或容器环境中。这种方法将配置与应用程序逻辑分离，使你的 Docker 镜像更易于迁移和适应。

### 为什么要使用环境变量进行配置？

使用环境变量具有以下几个重要优势：

1. **跨环境的灵活性：** 你可以构建一个 Docker 镜像，并通过在启动容器时设置不同的环境变量，将其部署到不同的环境（开发、测试、生产）。仅仅为了调整配置，不需要修改代码或重新构建镜像。
2. **提升安全性：** 敏感信息，如 API 密钥、数据库密码或秘密密钥，绝不应硬编码在源代码中或直接集成到 Docker 镜像中。环境变量提供了一种在运行时将这些秘密注入容器的机制，使它们不会出现在版本控制和镜像层中。
3. **遵循十二要素应用原则：** 将配置存储在环境中是[十二要素应用方法论](https://12factor.net/config)的一项基本原则，该方法论是一套被广泛认可的、用于构建现代可伸缩 Web 应用程序的最佳实践。

### 在 Python 中访问环境变量

Python 的标准库提供了 `os` 模块，用于与操作系统交互，包括访问环境变量。读取它们的主要方式是使用 `os.environ`，其行为类似于一个字典。

```python
import os

# 访问环境变量
# 如果变量未设置，使用 os.environ['VAR_NAME'] 将引发 KeyError。
# api_key = os.environ['MY_API_KEY']

# 一种更安全的方法：使用 .get() 并提供可选的默认值
api_key = os.environ.get('MY_API_KEY', 'default_key_if_not_set')
model_path = os.environ.get('MODEL_PATH', './models/default_model.joblib') # 机器学习模型路径示例
log_level = os.environ.get('LOG_LEVEL', 'INFO')

print(f"API Key: {api_key}")
print(f"Model Path: {model_path}")
print(f"Log Level: {log_level}")
```

通常建议使用 `os.environ.get('VAR_NAME', default_value)` 或 `os.getenv('VAR_NAME', default_value)`，因为它允许你在环境变量未设置时提供默认值，从而防止应用程序因缺少配置而崩溃。

### 使用 Pydantic Settings 进行结构化配置

尽管 `os.environ.get` 在简单情况下效果良好，但管理大量配置变量可能会变得繁琐。FastAPI 集成了 Pydantic，它提供了一种强大的方法来管理应用程序设置，包括自动从环境变量中读取。

Pydantic 的设置管理（现在常通过 `pydantic-settings` 库使用）允许你使用 Pydantic 模型定义配置模式。Pydantic 会尝试从环境变量中加载模型字段的值，并自动执行类型转换和验证。

首先，确保你已安装 `pydantic-settings`：

```bash
pip install pydantic-settings
```

现在，你可以定义一个设置类：

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class AppSettings(BaseSettings):
    # Pydantic 将自动尝试从环境变量加载这些值
    # 示例：APP_TITLE 将从环境变量 APP_TITLE 中加载
    app_title: str = "ML Model API"
    log_level: str = "INFO"
    model_path: str = "./models/default_model.joblib"
    api_key: Optional[str] = None # 可选秘密信息的示例

    # 配置 Pydantic 设置
    model_config = SettingsConfigDict(
        # 环境变量通常为大写，但 Pydantic 默认不区分大小写
        case_sensitive=False,
        # 在开发期间，你可以选择从 .env 文件加载（需要 python-dotenv）
        # env_file = '.env'
    )

# 创建一个在整个应用程序中使用的单一实例
settings = AppSettings()
```

在此示例中：

- `AppSettings` 继承自 `pydantic_settings.BaseSettings`。
- 类中定义的字段（`app_title`、`log_level`、`model_path`、`api_key`）代表应用程序的配置设置。
- Pydantic 会尝试查找与字段名称匹配的环境变量（默认不区分大小写）。例如，它会查找 `APP_TITLE`、`LOG_LEVEL`、`MODEL_PATH` 和 `API_KEY`。
- 默认值直接在类定义中提供。
- 类型提示（例如 `str`、`Optional[str]`）确保加载的值被正确解析和验证。如果环境变量存在但无法转换为指定类型（例如，为 `int` 字段提供“not-an-integer”），Pydantic 将引发验证错误。

然后，你可以在 FastAPI 应用程序中需要配置值的地方导入并使用 `settings` 实例，通常是使用依赖注入：

```python
# main.py（或你的相关路由文件）
from fastapi import FastAPI, Depends
from .config import AppSettings, settings # 导入实例

# 获取设置的依赖函数
def get_settings() -> AppSettings:
    return settings

app = FastAPI()

@app.get("/info")
async def info(current_settings: AppSettings = Depends(get_settings)):
    # 通过依赖访问设置
    return {
        "app_title": current_settings.app_title,
        "log_level": current_settings.log_level,
        "model_path_configured": current_settings.model_path
    }

# 如果不使用 Depends，你也可以直接使用 settings 实例
print(f"Starting application: {settings.app_title}")
# ... 应用程序的其他设置
```

使用 Pydantic `BaseSettings` 提供了一种结构化、已验证且类型安全的方式来处理从环境中加载的配置。

### 为 Docker 容器设置环境变量

既然你的应用程序知道如何读取环境变量，那么如何将它们提供给 Docker 容器呢？有几种常见方法：

1. **在 Dockerfile 中使用 `ENV` 指令：**
   你可以在 `Dockerfile` 中直接设置默认环境变量。这些值会被构建到镜像中。这适用于非敏感的默认值或不太可能在不同环境之间频繁更改的变量。

   ```dockerfile
   # Dockerfile 片段
   FROM python:3.9-slim

   WORKDIR /app

   COPY ./requirements.txt /app/requirements.txt
   RUN pip install --no-cache-dir -r requirements.txt

   COPY ./app /app/app
   COPY ./models /app/models # 假设模型已被复制

   # 设置默认环境变量
   ENV LOG_LEVEL="INFO"
   ENV MODEL_PATH="/app/models/iris_model.joblib"
   ENV APP_TITLE="Default ML API Title"

   # 暴露 FastAPI 将运行的端口
   EXPOSE 8000

   # 使用 Uvicorn 运行应用程序的命令
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

   使用 `ENV` 设置的变量在镜像构建过程中可用，并且在容器从该镜像启动时作为默认值。
2. **使用 `docker run -e` 或 `--env` 标志：**
   在启动容器时，你可以使用 `-e` 或 `--env` 标志来覆盖 `ENV` 的默认值或提供额外的环境变量。这是在运行时提供特定于环境的配置或秘密信息的最常用方法。

   ```bash
   # 运行容器，覆盖 LOG_LEVEL 并设置特定的 API_KEY
   docker run -d -p 8000:8000 \
     -e LOG_LEVEL="DEBUG" \
     -e API_KEY="secret-prod-api-key-12345" \
     -e APP_TITLE="Production ML Service" \
     your-fastapi-image-name:latest
   ```

   每个 `-e` 标志设置一个环境变量（`VAR_NAME=value`）。这种方法使敏感数据不直接存在于镜像中。
3. **使用环境变量文件（`--env-file`）：**
   为了管理大量变量，尤其是在开发期间或与 Docker Compose 等工具一起使用时，你可以将它们放在一个文件（例如 `.env`）中，并将文件路径传递给 `docker run` 命令。

   ```bash
   # .env 文件示例（例如 config.env）
   LOG_LEVEL=DEBUG
   API_KEY=local-dev-key-abcdef
   MODEL_PATH=/app/models/dev_model.joblib
   APP_TITLE=Development ML API
   ```

   ```bash
   # 使用环境变量文件运行容器
   docker run -d -p 8000:8000 --env-file ./config.env your-fastapi-image-name:latest
   ```

   Docker 会将文件中的每一行读取为 `KEY=VALUE` 对，并在容器内设置相应的环境变量。通过 `-e` 设置的变量通常会覆盖 `--env-file` 中的变量。

### 处理敏感信息

尽管环境变量相对于硬编码秘密信息是一个大幅改进，但请注意，在某些环境中，它们可能仍然通过容器检查工具或日志可见。对于高度敏感的生产秘密，请考虑使用专用的秘密管理系统（如 HashiCorp Vault、AWS Secrets Manager、Google Secret Manager、Azure Key Vault）。这些系统提供更多的安全特性，例如访问控制、审计和秘密轮换。集成这些系统通常涉及在应用程序启动时获取秘密或使用 sidecar 容器，这是一种更高级的部署模式，超出了本节的即时范围，但对于生产环境的安全加固需要了解。然而，对于许多应用程序而言，谨慎管理的环境变量在安全性和简单性之间提供了良好的平衡。

通过运用环境变量，特别是当与 Pydantic 的设置管理结合使用时，你可以创建灵活、可移植且更安全的容器化 FastAPI 应用程序，以适应不同的部署阶段。这种做法对于构建能够轻松适应其运行环境的应用程序非常重要。

## 参考资料

- [os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html) — Python Software Foundation (2024)
  Python `os`模块的官方文档，详细介绍了如何与操作系统交互并访问环境变量。
- [Settings management](https://docs.pydantic.dev/latest/usage/settings/) — Pydantic Development Team (2024)
  Pydantic-settings 的官方文档，解释了结构化配置管理，包括从环境变量加载带类型验证的设置。
- [Set environment variables (-e, --env, --env-file)](https://docs.docker.com/engine/reference/commandline/run/#set-environment-variables--e---env---env-file) — Docker, Inc. (2024)
  Docker 官方文档，关于如何使用 `docker run -e` 和 `--env-file` 管理容器的环境变量，这对于容器化应用配置至关重要。
