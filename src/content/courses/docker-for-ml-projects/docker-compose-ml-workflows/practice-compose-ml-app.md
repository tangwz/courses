---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "practice-compose-ml-app"
sourceId: 5353
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/practice-compose-ml-app"
title: "动手练习：使用 Compose 构建机器学习应用"
description: "编写 `docker-compose.yml` 文件以启动多容器应用（例如，推理服务和消息队列）。"
order: 8
plots: []
sourceHash: "ead1488bb267004a993f0098b0119193708c2292fbf39a95b19cb02981f5b39a"
sourceCorrections: []
---

一个简单的多容器机器学习 (machine learning)应用将使用 Docker Compose 构建。该应用包含两个服务：

1. **推理 (inference) API：** 一个简单的网络服务（使用 Flask），用于加载一个模拟模型并提供预测。
2. **Redis 缓存：** 一个 Redis 实例，供 API 用于缓存预测结果，展示容器间通信。

“这种设置反映了常见情景，即机器学习服务可能依赖于数据库或缓存等其他后端组件。”

### 项目设置

首先，为我们的项目创建一个目录，命名为 `compose_ml_app`。在此目录中，创建以下结构：

```
compose_ml_app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

- `app/`：此目录将包含我们的 Flask API 代码。
- `app/main.py`：Flask 应用的主 Python 脚本。
- `app/requirements.txt`：列出 API 的 Python 依赖项。
- `Dockerfile`：定义如何为我们的推理 (inference) API 服务构建镜像。
- `docker-compose.yml`：定义多容器应用栈。

### 构建推理 (inference) API

让我们为 Flask API 服务创建组件。

1. **`app/requirements.txt`**：
   列出必需的 Python 库。在本例中，我们需要 Flask 来创建 API，以及 `redis-py` 来与 Redis 缓存交互。

   ```
   Flask==2.3.3
   redis==5.0.1
   ```

   *注意：使用特定版本有助于提高可复现性。*
2. **`app/main.py`**：
   此脚本设置了一个基本的 Flask 应用，带有一个 `/predict` 端点。它模拟加载模型并进行预测。在预测之前，它会检查给定输入数据的结果是否已存在于 Redis 缓存中。如果是，则返回缓存结果；否则，它计算结果，将其存储在缓存中，然后返回。

   ```python
   import os
   import time
   import redis
   from flask import Flask, request, jsonify

   app = Flask(__name__)

   # 连接到 Redis - 使用 docker-compose.yml 中定义的 'redis' 服务名称
   # Docker Compose 自动提供同一网络中服务之间的 DNS 解析。
   redis_host = os.environ.get('REDIS_HOST', 'localhost')
   redis_port = int(os.environ.get('REDIS_PORT', 6379))
   cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0, decode_responses=True)

   # 模拟模型加载（如果需要，请替换为实际模型加载）
   def load_model():
       print("Simulating model loading...")
       time.sleep(2) # 模拟加载所需时间
       print("Model loaded.")
       # 在实际应用中，您将在此处加载您的 scikit-learn、TensorFlow、PyTorch 模型
       return "dummy_model"

   model = load_model()

   @app.route('/predict', methods=['POST'])
   def predict():
       try:
           data = request.get_json()
           if not data or 'input' not in data:
               return jsonify({"error": "Invalid input. Expecting JSON with 'input'."}), 400

           input_data = str(data['input']) # 使用输入作为缓存键

           # 首先检查缓存
           cached_result = cache.get(input_data)
           if cached_result:
               print(f"Cache hit for input: {input_data}")
               return jsonify({"prediction": cached_result, "source": "cache"})

           # 如果不在缓存中，则模拟预测
           print(f"Cache miss for input: {input_data}. Predicting...")
           # 将此替换为您实际的 model.predict() 调用
           time.sleep(0.5) # 模拟预测时间
           prediction_result = f"prediction_for_{input_data}"

           # 将结果存储在缓存中，并设置过期时间（例如，1 小时）
           cache.setex(input_data, 3600, prediction_result)
           print(f"Stored prediction in cache for input: {input_data}")

           return jsonify({"prediction": prediction_result, "source": "computation"})

       except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}")
            return jsonify({"error": "Could not connect to cache service."}), 503
       except Exception as e:
           print(f"An error occurred: {e}")
           return jsonify({"error": "An internal error occurred."}), 500

   @app.route('/health', methods=['GET'])
   def health_check():
       # 基本健康检查
       try:
           # 检查与 Redis 缓存的连接
            cache.ping()
            return jsonify({"status": "ok", "cache_connected": True}), 200
       except redis.exceptions.ConnectionError:
            return jsonify({"status": "degraded", "cache_connected": False}), 503

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
   ```

   注意 Redis 主机是如何从环境变量 `REDIS_HOST` 中获取的。我们将在 `docker-compose.yml` 文件中设置它。默认值为 `localhost`，用于 Docker 之外的潜在本地测试。我们还添加了一个基本的 `/health` 端点。
3. **`app/__init__.py`**：
   在 `app` 目录中创建一个空的 `__init__.py` 文件，使其成为一个 Python 包。您可以在 Linux/macOS 上使用 `touch app/__init__.py` 来完成此操作，或在 Windows 中创建一个名为 `__init__.py` 的空文件。

### 为 API 创建 Dockerfile

现在，在项目根目录 (`compose_ml_app/`) 中定义 `Dockerfile`，以便将 Flask 应用容器化。

```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置容器中的工作目录
WORKDIR /app

# 将 requirements 文件复制到容器的 /app 目录
COPY app/requirements.txt .

# 安装 requirements.txt 中指定的任何必需包
# 使用 --no-cache-dir 减少镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 将应用程序代码复制到容器的 /app 目录
COPY ./app /app

# 使端口 5000 在此容器外部可用
EXPOSE 5000

# 定义环境变量默认值（可由 docker-compose 覆盖）
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# 容器启动时运行 main.py
CMD ["python", "main.py"]
```

此 Dockerfile 设置了一个 Python 3.9 环境，安装依赖项，复制应用代码，暴露 Flask 将运行的端口，设置 Redis 连接的默认环境变量，并指定启动应用的命令。

### 使用 `docker-compose.yml` 定义多容器应用

Docker Compose 在此发挥作用。在项目根目录 (`compose_ml_app/`) 中创建 `docker-compose.yml` 文件。

```yaml
version: '3.8' # 指定 Compose 文件版本

services:
  # 服务 1：推理 API
  api:
    build: . # 从当前目录的 Dockerfile 构建镜像
    ports:
      - "5001:5000" # 将主机端口 5001 映射到容器端口 5000
    environment:
      - REDIS_HOST=redis # 覆盖默认值，确保其使用 redis 服务名称
      - REDIS_PORT=6379
    volumes:
      # 将本地 'app' 目录挂载到容器中的 '/app'
      # 这允许在开发过程中更改代码而无需重新构建镜像
      - ./app:/app
    depends_on:
      - redis # 确保 redis 在 api 服务启动之前启动
    networks:
      - ml_app_network # 将此服务附加到自定义网络

  # 服务 2：Redis 缓存
  redis:
    image: "redis:6.2-alpine" # 使用 Docker Hub 上的标准 Redis 镜像
    ports:
      - "6379:6379" # 暴露 Redis 端口，以便进行潜在的直接检查（可选）
    volumes:
      - redis_data:/data # 挂载一个命名卷以实现数据持久化
    networks:
      - ml_app_network # 将此服务附加到自定义网络

# 定义命名卷
volumes:
  redis_data: # 即使容器被移除，Redis 数据也会持久存在

# 定义自定义网络
networks:
  ml_app_network:
    driver: bridge # 使用默认的 bridge 驱动
```

让我们来解析这个 `docker-compose.yml` 文件：

- `version: '3.8'`：指定 Compose 文件格式的版本。
- `services:`：定义构成我们应用的容器。
  - `api:`：定义我们的 Flask 应用服务。
    - `build: .`：告诉 Compose 使用当前目录中的 `Dockerfile` 构建镜像。
    - `ports: - "5001:5000"`：将宿主机上的端口 5001 映射到 `api` 容器内部的端口 5000（Flask 运行的端口）。我们在宿主机上使用 5001 端口，以避免与在 5000 端口运行的其他服务发生潜在冲突。
    - `environment:`：在 `api` 容器内设置环境变量。重要的是，`REDIS_HOST=redis` 告诉我们的 Flask 应用连接到名为 `redis` 的服务。Compose 处理 DNS 解析。
    - `volumes: - ./app:/app`：将本地 `app` 目录挂载到容器的 `/app` 目录中。在 `app/` 中对 Python 代码进行的本地更改将立即反映到运行中的容器中，这对于开发非常有帮助。
    - `depends_on: - redis`：指定 `api` 服务依赖于 `redis` 服务。Compose 会先启动 `redis`，并等待其运行后再启动 `api`。*注意：`depends_on` 只等待容器启动，不一定等待容器内的应用程序就绪。更多检查通常涉及健康检查或等待脚本。*
    - `networks: - ml_app_network`：将服务连接到我们定义的网络。
  - `redis:`：定义我们的 Redis 缓存服务。
    - `image: "redis:6.2-alpine"`：告诉 Compose 从 Docker Hub 拉取指定的 Redis 镜像。`alpine` 标签表示这是一个更小的镜像变体。
    - `ports: - "6379:6379"`：如果需要调试，可以选择将标准 Redis 端口映射到主机，以便直接访问。
    - `volumes: - redis_data:/data`：将命名卷 `redis_data` 挂载到 Redis 容器内部的 `/data` 目录，这是 Redis 存储数据的地方。这确保了即使 `redis` 容器停止并重新启动，缓存的数据也会持久存在。
    - `networks: - ml_app_network`：将服务连接到我们定义的网络。
- `volumes:`：声明命名卷 `redis_data`。Docker 管理此卷。
- `networks:`：声明一个自定义桥接网络 `ml_app_network`。使用自定义网络是推荐的做法，提供更好的隔离性，并实现服务名称（`api`、`redis`）之间的 DNS 解析。

### 运行应用

在终端中导航到您的 `compose_ml_app` 目录。

1. **构建并启动服务：**
   运行以下命令：

   ```bash
   docker compose up --build
   ```

   - `docker compose up`：此命令读取 `docker-compose.yml` 文件，创建网络和卷（如果它们不存在），为 `api` 服务构建镜像（因为 `build: .`），拉取 `redis` 镜像，并启动这两个容器。
   - `--build`：强制 Docker Compose 为使用 `build` 指令定义的服务构建镜像，即使存在同名镜像也如此。当您更改了 `Dockerfile` 或应用代码依赖项时很有用。

   您将看到 `api` 和 `redis` 容器的日志在终端中交错显示。查找指示 Redis 已就绪和 Flask 应用正在运行的行（例如，`* Running on http://0.0.0.0:5000/`）。
2. **测试 API：**
   打开另一个终端窗口，使用 `curl`（或任何 HTTP 客户端）向您的 API 服务的 `/predict` 端点发送 POST 请求，该端点可在您的宿主机上的端口 5001 访问：

   ```bash
   # 第一次请求（缓存未命中）
   curl -X POST -H "Content-Type: application/json" \
        -d '{"input": "some_feature_vector_1"}' \
        http://localhost:5001/predict

   # 预期输出（可能略有不同）：
   # {"prediction":"prediction_for_some_feature_vector_1","source":"computation"}
   ```

   检查 `docker compose up` 的日志。您应该看到来自 Flask 应用的消息，指示“缓存未命中”，然后存储结果。

   现在，再次发送*相同*的请求：

   ```bash
   # 第二次请求（缓存命中）
   curl -X POST -H "Content-Type: application/json" \
        -d '{"input": "some_feature_vector_1"}' \
        http://localhost:5001/predict

   # 预期输出：
   # {"prediction":"prediction_for_some_feature_vector_1","source":"cache"}
   ```

   这次，响应指示结果来自“缓存”，日志应显示“缓存命中”消息。这证实了 `api` 服务通过 Compose 定义的 Docker 网络成功地与 `redis` 服务进行了通信。

   您也可以测试健康检查端点：

   ```bash
   curl http://localhost:5001/health
   # 预期输出：
   # {"cache_connected":true,"status":"ok"}
   ```
3. **停止应用：**
   返回运行 `docker compose up` 的终端并按 `Ctrl+C`。这将停止容器。

   为确保容器和网络被移除（命名卷 `redis_data` 默认会持久存在），运行：

   ```bash
   docker compose down
   ```

   如果您还想移除命名卷，请使用：

   ```bash
   docker compose down --volumes
   ```

### 可视化设置

我们可以用一个简单的图表来表示我们的 Compose 设置：

> 此图表显示客户端 (`curl`) 通过映射的宿主机端口与 `api` 服务交互。`api` 服务使用服务名称通过内部 Docker 网络 (`ml_app_network`) 与 `redis` 服务通信。`redis` 服务使用命名卷 (`redis_data`) 实现持久化。

这个动手练习展示了 Docker Compose 如何简化互联服务的管理。通过在 `docker-compose.yml` 中定义我们的应用栈，我们自动化了网络、卷、构建和容器启动顺序的创建，使我们的开发工作流程比管理单独的 `docker run` 命令更高效且可复现。您可以将此模式应用于更复杂的机器学习 (machine learning)应用，其中可能涉及数据库、消息队列、监控工具或其他组件。

## 参考资料

- [Docker Compose Documentation](https://docs.docker.com/compose/) — Docker (Current)
  Publisher: Docker
  这是理解和使用 Docker Compose 的主要资源，涵盖其功能、语法以及定义多容器应用程序的方法。
- [Docker Documentation](https://docs.docker.com/) — Docker (Current)
  Publisher: Docker
  提供 Docker 基础知识、容器化、使用 Dockerfile 构建镜像和网络方面的全面指南，这些是使用 Docker Compose 进行机器学习项目的基础。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications
  这本书解释了 Docker、容器化以及使用 Docker Compose 编排多容器应用程序，对实际的机器学习部署有帮助。
- [Flask Documentation](https://flask.palletsprojects.com/en/latest/) — Pallets (Current)
  Publisher: Pallets
  Flask Web 框架的官方指南，本实践部分使用该框架构建推理 API。
- [Redis Documentation](https://redis.io/docs/) — Redis (Current)
  Publisher: Redis Labs
  提供关于 Redis 作为内存数据存储和缓存的详细信息，解释了其功能和在服务间通信中的用法。
