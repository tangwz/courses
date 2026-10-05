# 在 Docker 容器中运行应用

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/running-app-docker-container)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Docker 镜像可以将 Flask 预测服务及其所有依赖项打包成一个独立的蓝图。当作为容器运行时，这个蓝图便会变为现实。容器是 Docker 镜像的一个实时运行实例，它提供了一个应用程序在其内部执行的独立环境。

## 使用 `docker run` 启动您的容器

从镜像启动容器的基本命令是 `docker run`。它的基本结构如下：

```bash
docker run [OPTIONS] IMAGE[:TAG] [COMMAND] [ARG...]
```

- `OPTIONS`：修改容器行为的标志（我们稍后会介绍一些重要选项）。
- `IMAGE[:TAG]`：您要运行的镜像名称。如果您按照上一节操作，这可能是 `prediction-service` 之类的名称。如果未指定标签，Docker 默认为 `:latest` 标签。
- `COMMAND` 和 `ARG...`：在容器内部执行的可选命令和参数 (parameter)，它们会覆盖 Dockerfile 中指定的默认命令（例如 `CMD ["python", "app.py"]` 指令）。

让我们试着运行 Flask 应用的容器。如果您的镜像名为 `prediction-service`，一个简单的执行方式是：

```bash
# 这可能会在前台运行，并把 Flask 日志输出到您的终端
docker run prediction-service
```

您可能会看到 Flask 的输出，提示服务器已启动。但是，这样运行会占用您的终端。要停止它，您可以按下 `Ctrl+C`。对于一个 Web 服务，我们通常希望它在后台运行。

## 在分离模式下运行

要在后台（分离模式）运行容器，请使用 `-d` 选项：

```bash
docker run -d prediction-service
```

此命令会启动容器并立即返回您的终端提示符，同时打印一个长容器 ID。但是我们如何与在容器 *内部* 运行的 Flask 应用交互呢？默认情况下，容器端口无法从您的宿主机访问。我们需要进行映射。

## 端口映射以便访问

容器内的 Flask 应用正在监听一个特定端口（可能是在 Flask 应用或 Dockerfile 中配置的 5000 端口）。要从我们的宿主机浏览器或 `curl` 等工具访问它，我们需要将宿主机上的一个端口映射到容器的端口。这通过使用 `-p` 或 `--publish` 选项来完成，格式为 `host_port:container_port`。

例如，要将宿主机上的 5000 端口映射到容器内部的 5000 端口，您可以使用 `-p 5000:5000`。

> 此图说明了 `-p` 标志如何将宿主机上的端口连接到 Flask 应用在容器内部运行的端口。

## 为容器命名

Docker 会为容器分配随机名称（例如 `focused_turing`）。这些名称虽然功能正常，但不太容易记住。您可以使用 `--name` 选项分配一个特定名称，这会使管理容器变得更方便。

## 综合运用

让我们将这些选项组合起来运行 Flask 服务容器：

1. 在分离模式下运行（`-d`）。
2. 将宿主机上的 5000 端口发布到容器中的 5000 端口（`-p 5000:5000`）。
3. 给它一个方便记忆的名称，例如 `my-prediction-app`（`--name my-prediction-app`）。
4. 指定镜像名称（`prediction-service`）。

```bash
docker run -d -p 5000:5000 --name my-prediction-app prediction-service
```

如果成功，Docker 将打印容器 ID，并且容器将在后台运行。

## 验证正在运行的容器

您如何知道它确实在运行呢？ `docker ps` 命令会列出所有当前正在运行的容器：

```bash
docker ps
```

您应该会看到类似这样的输出（具体细节可能有所不同）：

```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                    NAMES
a1b2c3d4e5f6   prediction-service   "python app.py"          15 seconds ago   Up 14 seconds   0.0.0.0:5000->5000/tcp   my-prediction-app
```

这确认了：

- 您的容器（`my-prediction-app`）正在运行。
- 它基于 `prediction-service` 镜像。
- 宿主机上的 5000 端口（`0.0.0.0:5000`）已映射到容器内部的 5000 端口。

您还可以使用 `docker logs` 命令查看容器内部应用产生的日志：

```bash
docker logs my-prediction-app
```

这应该会显示您的 Flask 应用的启动消息。

## 访问您的服务

现在容器正在运行且端口已映射，您可以像在本地测试时一样向您的预测服务发送请求，但目标是您发布的 *宿主机* 端口（我们示例中是 5000）上的 `localhost`。

使用 `curl`（假设您的 API 在 `/predict` 端点期望 JSON 输入）：

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.5, 3.0, 4.5]}' \
     http://localhost:5000/predict
```

或者使用 Python 的 `requests` 库：

```python
import requests
import json

# 示例输入数据（根据您的模型需求调整）
data = {"features": [1.0, 2.5, 3.0, 4.5]}

# Docker 中运行的服务的 URL
url = "http://localhost:5000/predict" # 使用宿主机端口

try:
    response = requests.post(url, json=data)
    response.raise_for_status() # 对于错误的 HTTP 状态码（4xx 或 5xx）抛出异常

    prediction = response.json()
    print(f"预测响应: {prediction}")

except requests.exceptions.RequestException as e:
    print(f"连接服务时出错: {e}")
except json.JSONDecodeError:
    print(f"无法解码 JSON 响应: {response.text}")
```

您应该会收到模型提供的预测输出，这些输出由完全在独立的 Docker 容器内运行的 Flask 应用提供服务！

## 停止和清理

完成实验后，您可以停止并删除容器以释放资源。

**停止容器：** 使用 `docker stop` 命令，后跟容器的名称或 ID：

```bash
docker stop my-prediction-app
```

**删除容器：** 已停止的容器仍然存在。要完全删除它，请使用 `docker rm`：

```bash
docker rm my-prediction-app
```

您只能删除已停止的容器。如果您想一次性停止并删除，可以在 `docker rm` 命令中使用 `-f`（强制）标志，但通常更好的做法是先停止它。

要查看所有容器，包括已停止的，请使用 `docker ps -a`。定期删除不再需要的已停止容器是良好的维护习惯。

您现在已成功在 Docker 容器中运行了您的机器学习 (machine learning)预测服务，这使其可移植并确保了运行环境的一致性。这是可靠部署应用的一个基础步骤。

## 参考资料

- [docker run](https://docs.docker.com/reference/cli/docker/run/) — Docker Docs (2024)
  此官方参考资料详细说明了 `docker run` 命令及其各种选项，例如分离模式（-d）、端口发布（-p）和命名（--name），这些对于容器执行至关重要。
- [Get started with Docker containers](https://docs.docker.com/get-started/overview/) — Docker Docs (2024)
  提供对 Docker 概念的基础理解，包括镜像和容器之间的区别，以及运行应用程序的基本步骤。
- [Docker in Action, Second Edition](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff and Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本实用书籍，解释了 Docker 基础知识、容器管理和应用程序的部署策略。

---

[上一节](06-%E4%B8%BAFlask%E5%BA%94%E7%94%A8%E6%9E%84%E5%BB%BADocker%E9%95%9C%E5%83%8F.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1.md)
