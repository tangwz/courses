# Compose 中的环境变量

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/compose-environment-variables)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Docker Compose 编排多容器应用时，有效的配置是必需的。堆栈中的服务，例如需要数据库凭据的推理 (inference) API 或需要特定超参数 (parameter) (hyperparameter)的训练作业，通常需要外部设置才能正常运行。针对每个容器手动通过复杂的 `docker run` 命令传递这些参数很快就会变得不切实际。

Docker Compose 提供了一种结构化且方便的方式来通过环境变量管理服务配置，可以直接在 `docker-compose.yml` 文件中定义，也可以通过外部文件引入。这种方法集中了配置，使您的多容器机器学习 (machine learning)应用更易于管理、调整和在不同环境中部署。

### 在 `docker-compose.yml` 中定义环境变量

设置服务环境变量最直接的方法是在 `docker-compose.yml` 中其定义中使用 `environment` 键。您可以将这些变量定义为列表或映射（字典）形式。

**使用列表形式：**

这通常是首选格式，因为它很明确，并且避免了潜在的类型解释问题。列表中的每个项都是 `VARIABLE=value` 格式的字符串。

```yaml
# docker-compose.yml
version: '3.8'
services:
  training_job:
    build: ./training_app
    image: my_training_app:latest
    environment:
      - LEARNING_RATE=0.01
      - EPOCHS=50
      - MODEL_OUTPUT_DIR=/app/outputs
      - WANDB_API_KEY=${WANDB_API_KEY_HOST} # 从宿主环境获取值
```

在此示例中，`LEARNING_RATE`、`EPOCHS` 和 `MODEL_OUTPUT_DIR` 是直接设置的。`WANDB_API_KEY` 展示了如何从运行 `docker-compose up` 的宿主机的环境中替换值。如果 `WANDB_API_KEY_HOST` 在您的宿主机上已设置，其值将被分配给容器内的 `WANDB_API_KEY`。如果宿主机上未设置，容器内的 `WANDB_API_KEY` 变量将为空。

**使用映射形式：**

另外，您可以使用字典格式。

```yaml
# docker-compose.yml
version: '3.8'
services:
  inference_api:
    build: ./inference_api
    image: my_inference_api:latest
    environment:
      MODEL_PATH: /models/latest.pkl
      API_PORT: 8000
      DATABASE_URL: ${DB_CONNECTION_STRING} # 从宿主环境获取值
    ports:
      - "8000:8000" # 将宿主端口 8000 映射到容器端口 8000
```

此处，`MODEL_PATH` 和 `API_PORT` 已设置。请注意，Compose 通常会将映射格式中的所有值转换为字符串。如果您的应用期望数字类型（如端口号），它需要内部处理字符串到数字的转换。`DATABASE_URL` 再次显示了从宿主环境中的替换。

### 替换宿主环境变量

如上所示，Compose 允许您将宿主机环境中的变量注入到容器环境中。这对于敏感信息（如 API 密钥或数据库密码）或在开发、测试和生产环境之间变化的设置非常有用。

语法是 `${HOST_VARIABLE}`。如果 `HOST_VARIABLE` 在宿主机上存在，则使用其值。否则，行为取决于您是否提供默认值：

- `${HOST_VARIABLE}`：如果在宿主机上未设置，容器内的变量将为空字符串。
- `${HOST_VARIABLE:-default}`：如果在宿主机上未设置，容器内的变量将获得 `default` 值。
- `${HOST_VARIABLE:?error message}`：如果在宿主机上未设置，Compose 将停止并显示错误消息。

带默认值的示例：

```yaml
# docker-compose.yml
services:
  worker:
    image: my_worker:latest
    environment:
      - QUEUE_NAME=${JOB_QUEUE:-default_queue}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
```

如果 `JOB_QUEUE` 在宿主机上未设置，容器内的 `QUEUE_NAME` 将是 `default_queue`。类似地，`LOG_LEVEL` 默认为 `INFO`。

### 使用 `.env` 文件进行本地开发

不建议将配置，尤其是秘密信息，直接硬编码到 `docker-compose.yml` 中。将此类文件提交到版本控制会暴露敏感数据。管理特定环境变量的常见做法，尤其是在本地开发期间，是使用一个环境文件，通常命名为 `.env`。

Docker Compose 会自动在您运行 `docker-compose up` 命令的目录（或使用 `--project-directory` 标志指定的项目目录）中查找名为 `.env` 的文件。定义在此 `.env` 文件中的变量将自动替换到您的 `docker-compose.yml` 文件中使用 `${VARIABLE}` 语法的地方。

**`.env` 文件示例：**

```bash
# .env（此文件通常应添加到 .gitignore 中）
DB_USER=ml_user
DB_PASSWORD=supersecretdevpassword
DB_HOST=db # 在 docker-compose.yml 中定义的服务名称
DB_NAME=mldatabase
API_KEY=dev-abc123xyz
```

**使用 `.env` 变量的 `docker-compose.yml` 示例：**

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: ./api
    image: my_ml_api:latest
    environment:
      - DATABASE_USER=${DB_USER}
      - DATABASE_PASSWORD=${DB_PASSWORD}
      - DATABASE_HOST=${DB_HOST}
      - DATABASE_NAME=${DB_NAME}
      - EXTERNAL_SERVICE_KEY=${API_KEY}
    depends_on:
      - db
    ports:
      - "5000:5000"

  db:
    image: postgres:14-alpine
    environment:
      # PostgreSQL 镜像使用这些特定环境变量
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

当您运行 `docker-compose up` 时，Compose 会读取 `.env` 文件，找到 `DB_USER=ml_user`，并将 `ml_user` 替换到 `docker-compose.yml` 文件中所有出现 `${DB_USER}` 的位置。这使得您的秘密信息不被版本控制系统追踪，同时为您的服务提供必要的配置。请记住将 `.env` 添加到您的 `.gitignore` 文件中！

### 使用 `env_file` 指定环境文件

尽管自动加载 `.env` 文件很方便，但有时您需要更多的控制，也许需要在不同情况下使用不同的环境文件（例如，`.env.dev`、`.env.prod`、`.env.test`）。服务定义中的 `env_file` 指令允许您指定一个或多个自定义环境文件。

```yaml
# docker-compose.yml
version: '3.8'
services:
  inference_service:
    build: ./inference
    image: my_predictor:latest
    # 首先从 ./config/.prod.env 加载变量，
    # 然后是 ./config/.common.env，覆盖重复项
    env_file:
      - ./config/.prod.env
      - ./config/.common.env
    environment:
      # 此处定义的变量将覆盖来自 env_file 的变量
      - LOG_LEVEL=INFO
```

通过 `env_file` 加载的变量会设置在容器的环境中。如果您列出多个文件，它们将按顺序读取，后面文件中定义的变量会覆盖前面文件中的变量。直接在 `environment` 键下定义的变量优先于通过 `env_file` 加载的变量。

### 优先级顺序

理解环境变量的应用顺序很重要：

1. 在 `docker-compose.yml` 中服务的 `environment` 部分定义的变量。
2. 通过命令行使用 `docker-compose run -e VARIABLE=value ...` 传递的变量（仅适用于 `run`）。
3. 使用 `env_file` 指令定义的变量。
4. 来自自动加载的 `.env` 文件的变量（用于 `docker-compose.yml` 中的替换）。
5. 在宿主环境中定义的变量（用于 `docker-compose.yml` 中的替换）。
6. 使用 Dockerfile 中的 `ENV` 指令定义的变量。

此列表中后设置的变量会被先设置的变量覆盖。

### 传递字面量美元符号

如果您需要容器内的环境变量实际包含美元符号（\$），您需要在 `docker-compose.yml` 文件中使用 `$$` 进行转义。

```yaml
# docker-compose.yml
services:
  my_service:
    image: some_image
    environment:
      - CONFIG_VAR=uses_a_literal_$$VAR # 在容器内部变为 CONFIG_VAR=uses_a_literal_$VAR
```

通过运用 `environment` 键、宿主变量替换、`.env` 文件以及 `env_file` 指令，Docker Compose 为配置您的多容器机器学习 (machine learning)应用提供了灵活且安全的方法。这使您能够将配置与应用程序代码和 Docker 镜像分离，使您的设置在不同开发和部署阶段更具可移植性且更易于管理。

## 参考资料

- [Environment variables in Compose](https://docs.docker.com/compose/environment-variables/) — Docker (2024)
  Publisher: Docker, Inc.
  官方文档，全面指导如何在 Docker Compose 中定义、替换和管理环境变量，包括 .env 文件和优先级规则。
- [Docker in Action, Third Edition](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications; Pages: 350
  本书介绍了 Docker 和 Docker Compose，其中包含关于服务配置和多容器设置中环境变量管理最佳实践的专章。

---

[上一节](04-%E5%9C%A8%20Compose%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E5%8D%B7.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20Compose%20%E6%9E%84%E5%BB%BA%E9%95%9C%E5%83%8F.md)
