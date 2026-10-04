---
course: "fastapi-ml-deployment"
chapter: "containerization-deployment-prep"
lesson: "production-deployment-gunicorn-uvicorn"
sourceId: 5355
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/production-deployment-gunicorn-uvicorn"
title: "生产部署准备 (Gunicorn/Uvicorn)"
description: "搭建使用 Gunicorn 进程管理器的生产级 ASGI 服务器，例如 Uvicorn。"
order: 7
plots: []
sourceHash: "f18fbc59818a08e40bfccea12a92b79c0433d52897d574ea528bccb3318d04f8"
sourceCorrections: []
---

虽然带 `--reload` 的 `uvicorn` 开发服务器在构建和迭代 FastAPI 应用时表现很好，但它并非为生产环境的需求而设计。生产环境要求稳定性、高效处理多个并发请求的能力以及抵御故障的能力。这就是生产级 ASGI 服务器和进程管理器的用武之地。

您已经在使用 Uvicorn，它是一个速度非常快的 ASGI（异步服务器网关接口）服务器，是运行 FastAPI 应用的根基。然而，在生产环境中直接使用简单的命令 `uvicorn main:app` 运行 Uvicorn 缺少进程管理功能。如果单个 Uvicorn 进程崩溃，您的应用就会下线。它本身也无法管理多个工作进程，以有效利用多核处理器来处理并发请求。

为解决这些问题，我们通常会引入一个进程管理器。在 Python 生态系统中，一个非常流行且经过考验的选择是 Gunicorn (Green Unicorn)。尽管 Gunicorn 本身主要是一个 WSGI 服务器，但通过使用专用的 Uvicorn 工作类，它也可以充当 ASGI 应用的进程管理器。

### 将 Gunicorn 与 Uvicorn 工作器配合使用

在生产环境中部署 FastAPI 应用的标准方法是使用 Gunicorn 来管理 Uvicorn 工作进程。Gunicorn 处理以下任务：

- 启动多个工作进程（您的应用实例）。
- 监控工作进程的健康状况，并在它们崩溃时重新启动。
- 在可用工作器之间分发传入请求。
- 优雅地处理服务器信号（例如，实现零停机重启）。

在此设置中，Gunicorn 监听传入的 HTTP 请求，并将其转发给其管理的某个 Uvicorn 工作器。然后，Uvicorn 工作器借助其高性能 ASGI 能力，通过您的 FastAPI 应用处理请求。

> Gunicorn 主进程管理多个 Uvicorn 工作器，每个工作器都运行一个 FastAPI 应用实例。

要使用此设置，首先确保您的环境中已安装 Gunicorn（或已添加到您的 `requirements.txt` 中）：

```bash
pip install gunicorn
```

然后，您可以使用类似以下命令来运行您的应用：

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000
```

让我们来解释一下这条命令：

- `gunicorn`：启动 Gunicorn 服务器的命令。
- `-w 4`：这指定要启动的工作进程数量。`4` 只是一个例子。一个常见的起始点是 `(2 * CPU 核心数) + 1`。然而，最佳数量在很大程度上取决于您的应用工作负载（CPU 密集型还是 I/O 密集型）以及您的 ML 模型推理 (inference)的性质。您需要进行试验并监控性能以找到最佳值。对于 CPU 密集型推理任务，工作器数量多于 CPU 核心数可能会因上下文 (context)切换而导致收益递减。
- `-k uvicorn.workers.UvicornWorker`：这很重要。它告诉 Gunicorn 使用 Uvicorn 的工作类，使 Gunicorn（一个 WSGI 服务器）能够管理并与 Uvicorn（一个 ASGI 服务器）工作器通信。
- `main:app`：这与 Uvicorn 使用的格式相同，指定 Python 模块 (`main`) 以及该模块内的 FastAPI 应用实例 (`app`)。根据您的项目结构进行调整。
- `-b 0.0.0.0:8000`：这将 Gunicorn 服务器绑定到在 `8000` 端口上监听所有可用网络接口 (`0.0.0.0`)。绑定到 `0.0.0.0` 对于 Docker 容器内的应用从容器外部进行访问是必需的。端口 `8000` 是典型端口，但可以更改。

### 与 Dockerfile 集成

现在您已经了解如何使用 Gunicorn 运行应用，您需要修改您的 `Dockerfile`，以便在容器启动时使用此命令。您将替换 `CMD` 指令，而不是使用开发用的 `uvicorn` 命令：

```dockerfile
# (假设之前的 Dockerfile 步骤：基本镜像、复制代码、安装依赖)
# ...

# 暴露 Gunicorn 将监听的端口
EXPOSE 8000

# 设置使用 Uvicorn 工作器运行 Gunicorn 的默认命令
# 使用环境变量增加灵活性 (见下文)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "-b", "0.0.0.0:8000"]
```

### 使用环境变量进行配置

将工作器数量或端口等值直接硬编码到 `Dockerfile` 的 `CMD` 中并非总是理想的。不同的部署环境（预生产、生产）可能需要不同的设置。正如上一节所讨论的，环境变量是处理此问题的标准方式。

您可以修改 `CMD` 以从环境变量中读取值。Gunicorn 允许通过环境变量或配置文件设置许多选项。例如，您可以在运行容器时通过环境变量设置工作器数量和端口：

```dockerfile
# (假设之前的 Dockerfile 步骤)
# ...

# 暴露默认端口 (可被覆盖)
EXPOSE 8000

# 使用环境变量定义默认值
ENV PORT=8000
ENV WORKERS=4
ENV APP_MODULE="main:app"

# 在命令中使用环境变量
# 注意：通常需要 CMD 的 shell 形式才能正确扩展变量
CMD gunicorn -w ${WORKERS} -k uvicorn.workers.UvicornWorker ${APP_MODULE} -b 0.0.0.0:${PORT}
```

现在，当您运行 Docker 容器时，可以覆盖这些默认值：

```bash
# 使用默认的 4 个工作器在 8000 端口运行
docker run -p 8000:8000 your-ml-api-image

# 使用 8 个工作器在 9000 端口运行
docker run -p 9000:9000 -e WORKERS=8 -e PORT=9000 your-ml-api-image
```

这种方法为在各种部署场景中配置您的应用服务器提供了更大的灵活性，而无需重新构建 Docker 镜像。

通过结合 Gunicorn 进行进程管理和 Uvicorn 工作器进行高性能 ASGI 请求处理，您创建了一个明显比基本开发服务器更稳定、可扩展且适合在生产环境中服务您的机器学习 (machine learning)模型的设置。这个容器化、可用于生产环境的服务器设置，是您可能将应用部署到云平台或内部基础设施之前的最后一步。

## 参考资料

- [Deployment - FastAPI](https://fastapi.tiangolo.com/deployment/) — Sebastián Ramírez (2024)
  FastAPI 官方生产部署指南，涵盖 Gunicorn 与 Uvicorn worker 的使用。
- [Gunicorn (Green Unicorn) Documentation](https://gunicorn.org/) — Benoit Chesneau, Tarek Ziadé, J. P. Calderone, et al. (2024)
  Gunicorn 的官方文档，详细介绍了其架构、worker 管理和生产环境配置。
- [Uvicorn Documentation](https://www.uvicorn.org/) — Tom Christie, Encode OSS (2024)
  Uvicorn 的官方文档，涵盖其作为 ASGI 服务器的作用以及用于 Gunicorn 的 UvicornWorker 类。
- [Docker Documentation](https://docs.docker.com/) — Docker, Inc. (2024)
  理解 Docker 概念、Dockerfile 和与部署相关的容器编排的官方来源。
